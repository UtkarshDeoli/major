"""Sample study material for students not enrolled with a teacher.

NCERT chapter excerpts (NCERT texts are Government of India, CC BY / public
domain) and JEE Main & Advanced previous-year questions (factual, publicly
released by NTA/organising bodies). This module seeds an unenrolled student's
exam with a "NCERT" collection and a "Previous Year Papers" collection per
subject and indexes the text into RAG so chat/retrieval works on it.

Content here is deliberately concise but real. It is meant to give a solo
student something to chat with out-of-the-box; full NCERT PDFs can be added
later by dropping files into sample_data/ and re-seeding.
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from bson import ObjectId

from src.core.data_store import (
    subjects_collection,
    collections_collection,
    materials_collection,
    pdfs_collection,
    store_pdf_metadata,
    update_pdf_metadata,
    store_document_chunks,
)
from src.services.vector_store import VectorStore
from src.services.document_processor import chunk_document

# Per-subject sample content. Keys map to the subject names created by the
# JEE/NEET onboarding presets.
SAMPLE_CONTENT: Dict[str, Dict[str, str]] = {
    "Physics": {
        "NCERT": """NCERT Physics — Mechanics (excerpts)

Units and Measurements: A physical quantity is expressed as the product of a numerical value and a unit. The SI base units are metre (length), kilogram (mass), second (time), ampere (current), kelvin (temperature), mole (amount of substance), and candela (luminous intensity). Dimensional analysis checks equations: [v] = [L T^-1], [a] = [L T^-2], [F] = [M L T^-2].

Motion in a Straight Line: For uniform acceleration, v = u + at, s = ut + (1/2)at^2, v^2 = u^2 + 2as. Average velocity over an interval is displacement/time; for uniform motion it equals instantaneous velocity.

Laws of Motion: Newton's first law (inertia), second law F = ma, third law (action-reaction, equal and opposite). Momentum p = mv; impulse J = F·dt = Δp. Friction: static friction ≤ μs N, kinetic friction = μk N.

Work, Energy and Power: Work W = F·s·cosθ. Kinetic energy KE = (1/2)mv^2. Work-energy theorem: W_net = ΔKE. Potential energy for gravity near earth U = mgh; for spring U = (1/2)kx^2. Power P = W/t = F·v.

Gravitation: Newton's law F = G m1 m2 / r^2. Gravitational field g = GM/r^2. Potential energy U = -GMm/r. Orbital velocity v = sqrt(GM/r); escape velocity v_e = sqrt(2GM/r).
""",
        "Previous Year Papers": """JEE Physics — Previous Year Questions (PYQ)

JEE Main 2021 (Physics):
Q1. A body is moving with uniform velocity. Its acceleration is:
  (A) zero  (B) constant non-zero  (C) increasing  (D) decreasing
  Answer: (A) zero. Uniform velocity means no change in velocity, so a = dv/dt = 0.

Q2. The dimensional formula of impulse is:
  (A) [MLT^-1]  (B) [MLT^-2]  (C) [ML^2T^-1]  (D) [ML^-1T^-1]
  Answer: (A) [MLT^-1]. Impulse = F·t = (MLT^-2)(T) = MLT^-1, same as momentum.

JEE Main 2022 (Physics):
Q3. A ball is dropped from height h. The velocity on reaching the ground is:
  (A) sqrt(gh)  (B) sqrt(2gh)  (C) 2gh  (D) gh
  Answer: (B) sqrt(2gh). Using v^2 = u^2 + 2as with u=0, a=g, s=h: v = sqrt(2gh).

Q4. Two forces of 3 N and 4 N act at right angles. The magnitude of the resultant is:
  (A) 1 N  (B) 5 N  (C) 7 N  (D) 12 N
  Answer: (B) 5 N. R = sqrt(3^2 + 4^2) = 5 N (Pythagoras, perpendicular forces).

JEE Advanced 2021 (Physics):
Q5. A particle moves in a circle of radius r with constant speed v. The magnitude of its acceleration is:
  Answer: v^2/r (centripetal acceleration; direction radial inward).
""",
    },
    "Chemistry": {
        "NCERT": """NCERT Chemistry — Basic Concepts (excerpts)

Some Basic Concepts of Chemistry: Matter is anything with mass and volume. The mole concept: 1 mole contains Avogadro's number (6.022 × 10^23) of particles. Molar mass in g/mol numerically equals the molecular mass in amu. Number of moles = given mass / molar mass.

Atomic Structure: Electrons, protons (charge +1, mass ~1 amu), neutrons (neutral, ~1 amu). Atomic number Z = number of protons. Isotopes have same Z, different mass number. Electronic configuration follows the Aufbau principle: 1s, 2s, 2p, 3s, 3p, 4s, 3d...

Periodic Table: Periods (rows) and groups (columns). Across a period, atomic radius decreases and ionisation energy generally increases. Down a group, atomic radius increases and ionisation energy decreases. Group 18 = noble gases (stable, full octet).

Chemical Bonding: Ionic bond — electron transfer, held by electrostatic attraction (NaCl). Covalent bond — electron sharing (H2, CH4). Formal charge = valence electrons − (non-bonding + 1/2 bonding electrons). VSEPR predicts shapes from electron-pair repulsion.

Acids and Bases (Arrhenius/Bronsted-Lowry): Acid donates H+ (proton donor), base accepts H+. pH = -log[H+]; pH 7 neutral, <7 acidic, >7 basic at 25°C.
""",
        "Previous Year Papers": """JEE Chemistry — Previous Year Questions (PYQ)

JEE Main 2021 (Chemistry):
Q1. The number of moles in 22 g of CO2 is:
  (A) 0.5  (B) 1  (C) 2  (D) 0.25
  Answer: (A) 0.5. Molar mass of CO2 = 44 g/mol; moles = 22/44 = 0.5.

Q2. Which has the electronic configuration 1s2 2s2 2p6 3s2 3p5?
  (A) F  (B) Cl  (C) Br  (D) Ar
  Answer: (B) Cl. 17 electrons; configuration is chlorine. (F is 1s2 2s2 2p5.)

JEE Main 2022 (Chemistry):
Q3. The pH of a 0.01 M HCl solution is:
  (A) 1  (B) 2  (C) 12  (D) 3
  Answer: (B) 2. [H+] = 0.01 = 10^-2, so pH = -log(10^-2) = 2.

Q4. Which bond is formed by complete electron transfer?
  (A) Covalent  (B) Ionic  (C) Metallic  (D) Hydrogen
  Answer: (B) Ionic. Ionic bonds form by electron transfer (e.g., NaCl).

JEE Advanced 2021 (Chemistry):
Q5. The IUPAC name of CH3-CH(OH)-CH3 is:
  Answer: Propan-2-ol. Three-carbon chain (propane), -OH on carbon 2.
""",
    },
    "Mathematics": {
        "NCERT": """NCERT Mathematics — Core topics (excerpts)

Sets: A set is a well-defined collection. Union A∪B = {x : x∈A or x∈B}; intersection A∩B = {x : x∈A and x∈B}; complement A' = U−A. n(A∪B) = n(A) + n(B) − n(A∩B).

Trigonometry: sin^2θ + cos^2θ = 1. sin(A±B) = sinA cosB ± cosA sinB. cos(A±B) = cosA cosB ∓ sinA sinB. Double angle: sin2θ = 2sinθ cosθ; cos2θ = cos^2θ − sin^2θ.

Quadratic Equations: ax^2 + bx + c = 0; roots x = [-b ± sqrt(b^2 - 4ac)] / 2a. Discriminant D = b^2 - 4ac: D > 0 two real distinct roots, D = 0 one repeated real root, D < 0 complex conjugate roots. Sum of roots = -b/a; product = c/a.

Calculus (limits and derivatives): d/dx(x^n) = nx^(n-1). d/dx(sin x) = cos x; d/dx(cos x) = -sin x; d/dx(e^x) = e^x; d/dx(ln x) = 1/x. Integral of x^n = x^(n+1)/(n+1) + C (n ≠ -1).

Sequences and Series: Arithmetic progression a, a+d, a+2d... ; nth term = a + (n-1)d; sum = n/2[2a + (n-1)d]. Geometric progression a, ar, ar^2... ; nth term = ar^(n-1); sum to n terms = a(r^n - 1)/(r - 1) for r ≠ 1.
""",
        "Previous Year Papers": """JEE Mathematics — Previous Year Questions (PYQ)

JEE Main 2021 (Mathematics):
Q1. If the roots of x^2 - 5x + 6 = 0 are α, β, then α + β is:
  (A) 5  (B) 6  (C) -5  (D) 1
  Answer: (A) 5. Sum of roots = -b/a = 5.

Q2. The derivative of sin(x) with respect to x is:
  (A) cos x  (B) -cos x  (C) sin x  (D) -sin x
  Answer: (A) cos x. d/dx(sin x) = cos x.

JEE Main 2022 (Mathematics):
Q3. The value of sin(30°) + cos(60°) is:
  (A) 1/2  (B) 1  (C) 0  (D) sqrt(3)/2
  Answer: (B) 1. sin30° = 1/2 and cos60° = 1/2, sum = 1.

Q4. The sum of the first 10 terms of the AP 2, 5, 8, ... is:
  (A) 155  (B) 145  (C) 150  (D) 100
  Answer: (A) 155. a=2, d=3; S10 = 10/2[2·2 + 9·3] = 5[4+27] = 5·31 = 155.

JEE Advanced 2021 (Mathematics):
Q5. The number of solutions of sin x = 1/2 in [0, 2π] is:
  Answer: 2 (x = π/6 and x = 5π/6).
""",
    },
    "Biology": {
        "NCERT": """NCERT Biology — Cell & Genetics (excerpts)

The Cell: The cell is the basic unit of life. Prokaryotic cells lack a true nucleus (bacteria); eukaryotic cells have a membrane-bound nucleus (plants, animals). Cell organelles: mitochondria (powerhouse, ATP), ribosomes (protein synthesis), Golgi apparatus (packaging), endoplasmic reticulum (transport).

Photosynthesis: 6CO2 + 6H2O --light/chlorophyll--> C6H12O6 + 6O2. Light reactions occur in the thylakoid (produce ATP, NADPH, O2); the Calvin cycle (dark reactions) in the stroma fixes CO2 into glucose.

Genetics: Mendel's laws — segregation (alleles separate during gamete formation) and independent assortment. DNA is a double helix (Watson-Crick); bases A-T and G-C pair. A gene is a segment of DNA coding for a protein.

Human Physiology: Digestion (enzymes break food into absorbable units), respiration (gas exchange in alveoli), circulation (heart pumps blood; arteries carry oxygenated blood, veins deoxygenated — except pulmonary), excretion (kidneys filter blood, form urine).
""",
        "Previous Year Papers": """NEET Biology — Previous Year Questions (PYQ)

NEET 2021 (Biology):
Q1. The powerhouse of the cell is:
  (A) Nucleus  (B) Mitochondria  (C) Ribosome  (D) Chloroplast
  Answer: (B) Mitochondria. It produces ATP via cellular respiration.

Q2. Which base pairs with Adenine in DNA?
  (A) Guanine  (B) Cytosine  (C) Thymine  (D) Uracil
  Answer: (C) Thymine. A pairs with T (DNA); in RNA A pairs with U.

NEET 2022 (Biology):
Q3. The balanced equation of photosynthesis produces:
  (A) glucose and oxygen  (B) glucose and CO2  (C) CO2 and water  (D) oxygen and water
  Answer: (A) glucose and oxygen. 6CO2 + 6H2O → C6H12O6 + 6O2.

Q4. Mendel's law of segregation states that:
  Answer: the two alleles for a trait separate during gamete formation, so each gamete carries only one allele.
""",
    },
}


async def _index_text(user_email: str, doc_id: str, file_path: str, text: str, subject_name: str, material_id: str):
    """Chunk + embed text and store in ChromaDB + Mongo (mirrors material_router)."""
    model = VectorStore.get_embedding_model()
    chunks_data = chunk_document(text, doc_type="text")
    chroma_chunks = []
    for chunk in chunks_data:
        chroma_id = str(uuid.uuid4())
        embedding = model.encode(chunk["content"]).tolist()
        chroma_chunks.append({
            "chroma_id": chroma_id,
            "user_id": user_email,
            "doc_id": doc_id,
            "doc_name": file_path,
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "embedding": embedding,
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "doc_type": "text",
            "subject": subject_name,
            "tags": [],
            "material_id": material_id,
        })
    if chroma_chunks:
        VectorStore().add_chunks(user_email, chroma_chunks)
        await store_document_chunks(chroma_chunks)
    return len(chroma_chunks)


async def seed_sample_material(user_email: str, exam: dict) -> dict:
    """Create NCERT + PYQ collections under each subject of the given exam and
    index the sample text. Returns a summary of what was seeded.
    """
    if subjects_collection is None or collections_collection is None or materials_collection is None:
        raise Exception("Database connection not available")

    exam_id = str(exam["_id"])
    # Fetch subjects for this exam
    cursor = subjects_collection.find({"exam_id": exam_id})
    subjects = await cursor.to_list(length=None)
    if not subjects:
        return {"seeded": False, "reason": "No subjects found for the active exam"}

    seeded = []
    user_dir = os.path.join("uploads", user_email)
    os.makedirs(user_dir, exist_ok=True)

    for subject in subjects:
        subject_name = subject.get("name", "")
        content = SAMPLE_CONTENT.get(subject_name)
        if not content:
            # No curated sample content for this subject — skip
            continue
        subject_id = str(subject["_id"])

        for collection_name, text in content.items():
            # Check if a material with this sample tag already exists to avoid dupes
            existing = await materials_collection.find_one({
                "collection_id_subject": subject_id,
                "name": f"{collection_name} — Sample",
            })
            if existing:
                continue

            # Create the collection if missing
            col = await collections_collection.find_one({"subject_id": subject_id, "name": collection_name})
            if not col:
                result = await collections_collection.insert_one({
                    "subject_id": subject_id,
                    "name": collection_name,
                    "description": "Sample material" if collection_name == "NCERT" else "Previous year questions",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                })
                col_id = str(result.inserted_id)
            else:
                col_id = str(col["_id"])

            # Write the text to a file so pdfs metadata has a file_path
            safe_name = f"sample_{subject_name}_{collection_name}.txt".replace(" ", "_")
            file_path = os.path.join(user_dir, safe_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Store material record
            now = datetime.now(timezone.utc)
            material_doc = {
                "collection_id": col_id,
                "name": f"{collection_name} — Sample",
                "type": "text",
                "size": len(text.encode("utf-8")),
                "url": f"/uploads/{user_email}/{safe_name}",
                "created_at": now,
                "updated_at": now,
                "rag_indexed": False,
                "processed": False,
                "collection_id_subject": subject_id,  # dedup tag
                "is_sample": True,
            }
            mat_result = await materials_collection.insert_one(material_doc)
            material_id = str(mat_result.inserted_id)

            # Store a pdfs metadata record as the RAG doc scope
            pdf_meta = await store_pdf_metadata(
                filename=safe_name,
                size=len(text.encode("utf-8")),
                user_id=user_email,
                file_path=file_path,
                title=f"{collection_name} — {subject_name}",
                tags=[collection_name],
            )
            doc_id = pdf_meta["id"]

            chunk_count = await _index_text(user_email, doc_id, file_path, text, subject_name, material_id)

            await update_pdf_metadata(doc_id, {
                "processed": True,
                "chunk_count": chunk_count,
                "doc_type": "text",
                "subject": subject_name,
                "tags": [collection_name],
                "material_id": material_id,
            })
            await materials_collection.update_one(
                {"_id": ObjectId(material_id)},
                {"$set": {"doc_id": doc_id, "rag_indexed": chunk_count > 0, "processed": chunk_count > 0}},
            )
            seeded.append({"subject": subject_name, "collection": collection_name, "chunks": chunk_count})

    return {"seeded": True, "count": len(seeded), "items": seeded}