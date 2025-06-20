import datetime

from sqlalchemy import JSON  # Добавлено для хранения JSON данных
from sqlalchemy import Integer  # Добавлено для порядковых номеров
from sqlalchemy import Text  # Добавлено для более длинных текстовых полей
from sqlalchemy import (
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


class Student(Base, TimestampMixin):  # Переименовано из User
    """Модель студента."""

    __tablename__ = "students"  # Таблица переименована

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
    first_name = Column(String, nullable=True, comment="Имя студента")
    last_name = Column(String, nullable=True, comment="Фамилия студента")

    # Новые поля для студента
    course = Column(Integer, nullable=True, comment="Курс обучения")
    group = Column(String, nullable=True, comment="Номер группы")
    specialization = Column(String, nullable=True, comment="Специализация")
    enrollment_date = Column(DateTime, nullable=True, comment="Дата зачисления")

    is_admin = Column(
        Boolean,
        default=False,
        nullable=False,
        server_default="false",
        comment="Является ли пользователь администратором (True/False)",
    )

    # Связь с загруженными документами (если администратор - это тоже студент)
    uploaded_documents = relationship(
        "Document",
        foreign_keys="[Document.uploaded_by_tg_id]",  # Явно указываем foreign_keys
        back_populates="uploaded_by_user",
        cascade="all, delete-orphan",
    )
    # Связь с прогрессом студента
    progress_records = relationship(
        "StudentProgress", back_populates="student", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Student(user_id={self.user_id}, username='{self.username}', is_admin={self.is_admin})>"


class Document(Base, TimestampMixin):
    """Модель документа (источника контента), загруженного в систему."""

    __tablename__ = "documents"

    lightrag_id = Column(
        String,
        primary_key=True,
        index=True,
        comment="Уникальный ID документа, используемый в LightRAG и как source_reference",
    )
    file_name = Column(
        String, nullable=False, comment="Оригинальное имя загруженного файла"
    )
    status = Column(
        String,
        nullable=False,
        index=True,
        default="pending",
        comment="Статус обработки документа (pending, processing, processed, failed, deleted_by_admin)",
    )
    uploaded_by_tg_id = Column(
        BigInteger, ForeignKey("students.user_id"), nullable=False
    )  # ForeignKey изменен на students.user_id
    uploaded_by_user = relationship(
        "Student", foreign_keys=[uploaded_by_tg_id], back_populates="uploaded_documents"
    )

    # Связь с темами учебного плана, которые ссылаются на этот документ
    curriculum_topics = relationship(
        "CurriculumTopic", back_populates="source_document"
    )

    def __repr__(self):
        return f"<Document(lightrag_id='{self.lightrag_id}', file_name='{self.file_name}', status='{self.status}')>"


class CurriculumTopic(Base, TimestampMixin):
    """Модель темы/главы/раздела учебного плана."""

    __tablename__ = "curriculum_topics"

    topic_id = Column(
        String,
        primary_key=True,
        index=True,
        comment="Уникальный идентификатор темы (может быть UUID или сгенерированный)",
    )
    topic_name = Column(String, nullable=False, comment="Название темы")

    parent_topic_id = Column(
        String,
        ForeignKey("curriculum_topics.topic_id"),
        nullable=True,
        index=True,
        comment="Ссылка на родительскую тему",
    )
    order = Column(
        Integer,
        nullable=True,
        comment="Порядок изучения внутри родительской темы или на верхнем уровне",
    )

    prerequisites = Column(
        JSON,
        nullable=True,
        comment="Список topic_id тем, которые нужно изучить до этой (например, ['topic_a', 'topic_b'])",
    )
    learning_objectives = Column(
        Text, nullable=True, comment="Чему должен научиться студент (описание)"
    )
    associated_tasks = Column(
        JSON,
        nullable=True,
        comment="Ссылки или описания связанных заданий, тестов (например, [{'type': 'quiz', 'id': 'quiz123'}])",
    )

    source_reference = Column(
        String,
        ForeignKey("documents.lightrag_id"),
        nullable=True,
        index=True,
        comment="ID документа-источника контента",
    )
    metadata = Column(
        JSON,
        nullable=True,
        comment="Дополнительная информация (автор, год, глава в источнике и т.д.)",
    )

    source_document = relationship("Document", back_populates="curriculum_topics")
    parent_topic = relationship(
        "CurriculumTopic", remote_side=[topic_id], back_populates="child_topics"
    )
    child_topics = relationship("CurriculumTopic", back_populates="parent_topic")

    progress_records = relationship("StudentProgress", back_populates="topic")

    def __repr__(self):
        return f"<CurriculumTopic(topic_id='{self.topic_id}', topic_name='{self.topic_name}')>"


class StudentProgress(Base, TimestampMixin):
    """Модель прогресса студента по темам."""

    __tablename__ = "student_progress"

    id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # Собственный первичный ключ для записи прогресса
    student_id = Column(
        BigInteger, ForeignKey("students.user_id"), nullable=False, index=True
    )
    topic_id = Column(
        String, ForeignKey("curriculum_topics.topic_id"), nullable=False, index=True
    )

    status = Column(
        String,
        nullable=False,
        default="not_started",
        comment="Статус изучения темы (e.g., 'not_started', 'in_progress', 'completed', 'needs_review')",
    )
    last_accessed = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        comment="Дата последнего обращения к теме",
    )
    assessment_results = Column(
        JSON,
        nullable=True,
        comment="Результаты тестов/опросов по теме (например, {'quiz123': {'score': 80, 'attempts': 2}})",
    )

    student = relationship("Student", back_populates="progress_records")
    topic = relationship("CurriculumTopic", back_populates="progress_records")

    __table_args__ = (
        Index(
            "idx_student_topic_progress", "student_id", "topic_id", unique=True
        ),  # Уникальный индекс для пары студент-тема
    )

    def __repr__(self):
        return f"<StudentProgress(student_id={self.student_id}, topic_id='{self.topic_id}', status='{self.status}')>"


# Существующие индексы, если они все еще актуальны или нужно адаптировать
Index("idx_document_status_uploaded_by", Document.status, Document.uploaded_by_tg_id)
Index("idx_student_is_admin", Student.is_admin)  # Изменено на Student.is_admin
