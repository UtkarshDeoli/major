import pytest
from src.core.models import Class, ClassSubject, ClassMaterial, ClassInvite
from src.core import data_store as ds


def test_class_supports_multi_teacher_and_org():
    c = Class(
        teacher_id="t@x.com",
        name="JEE 2026",
        enroll_code="ABC123",
        org_id="org-1",
        teacher_ids=["t@x.com", "t2@x.com"],
        subject_ids=[],
    )
    assert c.org_id == "org-1"
    assert c.teacher_ids == ["t@x.com", "t2@x.com"]
    assert c.subject_ids == []


def test_class_subject_model():
    s = ClassSubject(class_id="c1", name="Physics", created_by="t@x.com")
    assert s.class_id == "c1"
    assert s.name == "Physics"


def test_class_material_model():
    m = ClassMaterial(
        class_id="c1", class_subject_id="s1", teacher_id="t@x.com",
        name="notes.pdf", type="pdf", size=1024, doc_id="doc-1", rag_indexed=True,
    )
    assert m.doc_id == "doc-1" and m.rag_indexed is True


def test_class_invite_model():
    inv = ClassInvite(class_id="c1", email="stu@x.com", token="tok", status="pending", created_by="t@x.com")
    assert inv.status == "pending"


def test_new_collections_exist():
    # They are None only if MongoDB never connected; in the test env they may be None,
    # so we only assert the attributes are present (not left undefined).
    assert hasattr(ds, "class_subjects_collection")
    assert hasattr(ds, "class_materials_collection")
    assert hasattr(ds, "class_invites_collection")
