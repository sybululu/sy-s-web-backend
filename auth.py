"""
认证模块
=====================
用户注册、登录、JWT Token 管理
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr

from models import User, get_db

# ==========================================
# 配置 - SECRET_KEY 生成
# ==========================================
router = APIRouter(prefix="/api/v1/auth", tags=["认证"])

# 安全生成密钥（优先使用独立环境变量，其次使用 HF Spaces 内置变量，最后随机生成）
import secrets

# 1. 优先使用独立的 SECRET_KEY 环境变量
SECRET_KEY = os.environ.get("SECRET_KEY")

# 2. 如果没有独立密钥，从 HF Spaces 内置变量派生
if not SECRET_KEY:
    hf_space_id = os.environ.get("HF_SPACE_ID", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_space_id:
        # 使用 SPACE_ID + TOKEN 组合生成确定性密钥
        import hashlib
        combined = f"{hf_space_id}-{hf_token}"
        SECRET_KEY = hashlib.sha256(combined.encode()).hexdigest()
    elif hf_token:
        # 只有 TOKEN，使用 TOKEN 的哈希
        SECRET_KEY = hashlib.sha256(hf_token.encode()).hexdigest()
    else:
        # 完全没有可用的环境变量，生成随机密钥
        # 注意：这意味着每次重启服务 Token 都会失效（适用于开发）
        SECRET_KEY = secrets.token_hex(32)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ==========================================
# Pydantic 模型
# ==========================================
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class TokenResponse(BaseModel):
    token: str
    user: dict

class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None

# ==========================================
# 工具函数
# ==========================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """密码哈希"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建 JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ==========================================
# 依赖项
# ==========================================
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前用户"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="无效的 Token")
    except JWTError:
        raise HTTPException(status_code=401, detail="无效的 Token")
    
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    
    return user


async def get_current_user_from_token(token: str, db: Session) -> Optional[User]:
    """从 token 字符串获取用户（用于 query string 认证）"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        user = db.query(User).filter(User.id == int(user_id)).first()
        return user
    except JWTError:
        return None

# ==========================================
# API 端点
# ==========================================
@router.post("/login", response_model=TokenResponse)
async def login(
    req: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    用户登录
    
    返回:
    - token: JWT Token
    - user: 用户信息
    """
    user = db.query(User).filter(User.email == req.email).first()
    
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    
    # 更新最后登录时间
    user.last_login = datetime.utcnow()
    db.commit()
    
    # 生成 Token
    token = create_access_token({"sub": str(user.id)})
    
    return {
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name
        }
    }

@router.post("/register")
async def register(
    req: RegisterRequest,
    db: Session = Depends(get_db)
):
    """
    用户注册
    
    返回:
    - message: 注册成功消息
    - user_id: 用户ID
    """
    # 检查邮箱是否已注册
    existing_user = db.query(User).filter(User.email == req.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    
    # 创建用户
    user = User(
        email=req.email,
        password_hash=get_password_hash(req.password),
        name=req.name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "message": "注册成功",
        "user_id": user.id
    }

@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    """
    获取当前用户信息
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name
    }
