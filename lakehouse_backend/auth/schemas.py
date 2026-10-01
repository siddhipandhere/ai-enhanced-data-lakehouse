"""Pydantic request/response models for the auth API."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, description="Minimum 8 characters")


class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    username: str
    password: str
    # "Keep me signed in" on the login page -> longer-lived token
    remember_me: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int | None = None  # seconds until the token expires

