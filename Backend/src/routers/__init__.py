# This file makes the routers directory a Python package
from .auth_router import router as auth_router
from .pdf_router import router as pdf_router
from .document_router import router as document_router
from .question_router import router as question_router
from .analysis_router import router as analysis_router
from .mock_test_router import router as mock_test_router
from .teacher_router import router as teacher_router
from .analytics_router import router as analytics_router
from .exam_router import router as exam_router
from .subject_router import router as subject_router
from .collection_router import router as collection_router
from .material_router import router as material_router
from .onboarding_router import router as onboarding_router
from .flashcard_router import router as flashcard_router
from .ai_material_router import router as ai_material_router
from .class_router import router as class_router
from .sample_material_router import router as sample_material_router
