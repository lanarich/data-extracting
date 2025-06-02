import datetime

from sqlalchemy import (  # Добавлен Boolean
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class TimestampMixin:
    """Миксин для добавления временных меток created_at и updated_at."""

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
        server_onupdate=func.now(),
        nullable=False,
    )


class User(Base, TimestampMixin):
    """Модель пользователя Telegram."""

    __tablename__ = "users"

    user_id = Column(
        BigInteger, primary_key=True, index=True, comment="Telegram User ID"
    )
    username = Column(
        String,
        nullable=True,
        unique=True,
        index=True,
        comment="Telegram Username (@username)",
    )
    first_name = Column(String, nullable=True, comment="Имя пользователя")
    last_name = Column(String, nullable=True, comment="Фамилия пользователя")

    is_admin = Column(
        Boolean,
        default=False,
        nullable=False,
        server_default="false",
        comment="Является ли пользователь администратором (True/False)",
    )

    documents = relationship(
        "Document", back_populates="uploaded_by_user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}', is_admin={self.is_admin})>"


class Document(Base, TimestampMixin):
    """Модель документа, загруженного в систему."""

    __tablename__ = "documents"

    lightrag_id = Column(
        String,
        primary_key=True,
        index=True,
        comment="Уникальный ID документа, используемый в LightRAG",
    )
    file_name = Column(
        String, nullable=False, comment="Оригинальное имя загруженного файла"
    )

    status = Column(
        String,
        nullable=False,
        index=True,
        default="pending",
        comment="Статус документа (pending, processing, processed, failed, deleted_by_admin)",
    )

    uploaded_by_tg_id = Column(BigInteger, ForeignKey("users.user_id"), nullable=False)
    uploaded_by_user = relationship("User", back_populates="documents")

    def __repr__(self):
        return f"<Document(lightrag_id='{self.lightrag_id}', file_name='{self.file_name}', status='{self.status}')>"


Index("idx_document_status_uploaded_by", Document.status, Document.uploaded_by_tg_id)
Index("idx_user_is_admin", User.is_admin)  # Индекс для нового поля
