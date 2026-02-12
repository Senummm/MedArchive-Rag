"""
Download real, publicly available medical PDFs for testing the MedArchive RAG system.

Downloads from authoritative sources like CDC, WHO, and medical organizations.
All sources are publicly available and free to use for educational purposes.
"""

import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Publicly available medical PDFs from authoritative sources
MEDICAL_PDFS = [
    {
        "name": "Diabetes_Prevention_CDC.pdf",
        "url": "https://www.cdc.gov/diabetes/pdfs/prevent/On-your-way-to-preventing-type-2-diabetes.pdf",
        "description": "CDC Type 2 Diabetes Prevention Guide"
    },
    {
        "name": "Hypertension_Management_AHA.pdf",
        "url": "https://www.heart.org/-/media/Files/Professional/Quality-Improvement/Pain-Management/AHA-GUIDELINEDRIVEN-MANAGEMENT-OF-HYPERTENSION--AN-EVIDENCEBASED-UPDATE.pdf",
        "description": "AHA Guideline-Driven Hypertension Management"
    },
    {
        "name": "COVID19_Clinical_Guidelines.pdf",
        "url": "http://old.epid.gov.lk/web/images/pdf/Circulars/Corona_virus/covid-19%20cpg%20_%20version%204.pdf",
        "description": "COVID-19 Clinical Practice Guidelines"
    },
    {
        "name": "CVD_Prevention_Guidelines.pdf",
        "url": "https://www.ncd.health.gov.lk/images/pdf/circulars/National_Guideline_for_Risk_Assessment_and_Primary_Prevention_of_Cardiovascular_Diseases_2.pdf",
        "description": "Cardiovascular Disease Prevention Guidelines"
    },
    {
        "name": "Stroke_Guidelines_AHA.pdf",
        "url": "https://www.heart.org/-/media/Files/Professional/Quality-Improvement/Get-With-the-Guidelines/Get-With-The-Guidelines-Stroke/2019UpdateAHAASAAISGuidelineSlideDeckrevisedADL12919.pdf",
        "description": "AHA/ASA Acute Ischemic Stroke Guidelines 2019"
    },
]


def download_pdf(url: str, output_path: Path, description: str) -> bool:
    """
    Download a PDF from URL to output path.

    Args:
        url: URL to download from
        output_path: Local path to save PDF
        description: Description of the document

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading: {description}")
        print(f"   URL: {url}")

        # Set user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()

        # Verify it's a PDF
        if not content.startswith(b'%PDF'):
            print(f"   ⚠️  Warning: File may not be a valid PDF")
            return False

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(content)

        file_size = len(content) / 1024  # KB
        print(f"   ✅ Downloaded: {output_path.name} ({file_size:.1f} KB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"   ❌ HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"   ❌ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def create_simple_diabetes_pdf():
    """Create a simple diabetes complications PDF as fallback."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Diabetes Complications Overview", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)
    content = """
Diabetes Mellitus: Major Complications

1. Microvascular Complications:

Diabetic Retinopathy:
- Leading cause of blindness in working-age adults
- Caused by damage to retinal blood vessels
- Annual eye exams recommended for all diabetic patients
- Treatment includes laser therapy and anti-VEGF injections

Diabetic Nephropathy:
- Affects 30-40 percent of diabetic patients
- Leading cause of end-stage renal disease
- Screening: Annual urine albumin and serum creatinine
- Management: ACE inhibitors or ARBs, BP control, glycemic control

Diabetic Neuropathy:
- Affects up to 50 percent of patients with diabetes
- Distal symmetric polyneuropathy most common (feet and hands)
- Can also affect autonomic nervous system
- Treatment: Pain management, blood sugar control, foot care

2. Macrovascular Complications:

Cardiovascular Disease:
- 2-4 fold increased risk compared to non-diabetics
- Leading cause of death in diabetic patients
- Management: Aspirin, statins, BP control, SGLT2i or GLP-1 RA

Cerebrovascular Disease:
- 2-3 times higher stroke risk
- Prevention: BP control, antiplatelet therapy, statins

Peripheral Arterial Disease:
- Affects 20-30 percent of diabetic patients
- Increases amputation risk significantly
- Screening: Ankle-brachial index in patients over 50 years

3. Prevention Strategies:

Glycemic Control:
- HbA1c target less than 7 percent for most patients
- More stringent (less than 6.5 percent) for selected patients
- Regular blood glucose monitoring

Blood Pressure Management:
- Target less than 130 over 80 mmHg
- First-line: ACE inhibitors or ARBs

Lipid Management:
- Statin therapy for all patients 40-75 years
- LDL-C target less than 100 mg/dL (primary prevention)
- LDL-C target less than 70 mg/dL (secondary prevention)

Lifestyle Modifications:
- Mediterranean or DASH diet
- 150 minutes per week moderate exercise
- 7-10 percent weight loss if overweight
- Smoking cessation

4. Screening and Monitoring:

Annual Assessments:
- Comprehensive foot examination
- Dilated retinal exam
- Urine albumin-to-creatinine ratio
- Serum creatinine and eGFR
- Lipid panel
- Depression screening

Quarterly Monitoring:
- HbA1c testing (every 3 months)
- Blood pressure checks
- Weight and BMI

Patient Education:
- Sick day management
- Hypoglycemia recognition and treatment
- Proper foot care techniques
- Medication adherence importance
- Self-monitoring of blood glucose
"""

    pdf.multi_cell(0, 5, content.strip())

    output_path = project_root / "data" / "document_store" / "diabetes_complications_overview.pdf"
    pdf.output(str(output_path))
    return output_path


def create_simple_hypertension_pdf():
    """Create a simple hypertension treatment PDF as fallback."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Hypertension Treatment Guidelines", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)
    content = """
First-Line Treatment for Hypertension

1. Thiazide Diuretics:

Agents:
- Chlorthalidone 12.5-25 mg once daily (preferred)
- Hydrochlorothiazide 25-50 mg once daily
- Indapamide 1.25-2.5 mg once daily

Benefits:
- Effective BP reduction (10-15 mmHg systolic)
- Reduced cardiovascular events and stroke
- Low cost and well-tolerated

Monitoring: Electrolytes, creatinine, uric acid
Side Effects: Hypokalemia, hyponatremia, hyperuricemia

2. ACE Inhibitors:

Common Agents:
- Lisinopril 10-40 mg once daily
- Enalapril 5-40 mg once or twice daily
- Ramipril 2.5-20 mg once daily

Indications:
- First-line for diabetes, CKD, heart failure
- Post-myocardial infarction
- Proteinuria reduction

Contraindications: Pregnancy, angioedema history
Monitoring: Creatinine, potassium (within 2-4 weeks)
Side Effects: Dry cough (10 percent), hyperkalemia

3. Angiotensin Receptor Blockers (ARBs):

Common Agents:
- Losartan 50-100 mg once daily
- Valsartan 80-320 mg once daily
- Telmisartan 40-80 mg once daily

Indications:
- Alternative to ACE inhibitors (better tolerated)
- Diabetic nephropathy
- Heart failure with reduced EF

Benefits: Similar to ACE inhibitors without cough
Monitoring: Same as ACE inhibitors

4. Calcium Channel Blockers:

Dihydropyridines:
- Amlodipine 5-10 mg once daily
- Nifedipine XL 30-90 mg once daily

Non-dihydropyridines:
- Diltiazem CD 180-360 mg once daily
- Verapamil SR 120-480 mg once daily

Indications: Elderly patients, African Americans, CAD
Side Effects: Peripheral edema, headache

Treatment Algorithm:

Stage 1 HTN (130-139 over 80-89):
- Single agent plus lifestyle modifications
- Reassess in 3-6 months

Stage 2 HTN (greater than 140 over 90):
- Two agents from different classes
- Combinations: ACE or ARB plus CCB or ACE or ARB plus Thiazide

Resistant HTN (greater than 140 over 90 on 3 agents):
- Add spironolactone 25-50 mg daily
- Screen for secondary causes
- Consider specialist referral

Lifestyle Modifications:

DASH Diet:
- Rich in fruits, vegetables, whole grains
- Low-fat dairy, limited saturated fat
- Expected reduction: 8-14 mmHg

Sodium Restriction:
- Target less than 2,300 mg per day (less than 1,500 mg ideal)
- Expected reduction: 2-8 mmHg

Weight Loss:
- Target BMI less than 25 kg per meter squared
- Expected reduction: 5-20 mmHg per 10 kg loss

Exercise:
- 150 minutes per week moderate aerobic activity
- Expected reduction: 5-8 mmHg

Alcohol Moderation:
- Men 2 drinks or less per day, Women 1 drink or less per day
- Expected reduction: 2-4 mmHg

Monitoring Schedule:

Initial Phase:
- BP every 2-4 weeks until controlled
- Labs (BMP) within 2-4 weeks of med changes

Maintenance:
- BP every 3-6 months when stable
- Annual labs, ECG, urinalysis

Home BP Monitoring:
- Morning and evening measurements
- Two readings per session
- Record for 1 week quarterly
"""

    pdf.multi_cell(0, 5, content.strip())

    output_path = project_root / "data" / "document_store" / "hypertension_treatment_guide.pdf"
    pdf.output(str(output_path))
    return output_path


def main():
    """Main function to download and create PDFs."""
    print("\n🏥 Downloading Medical PDFs for MedArchive RAG System\n")
    print("=" * 70)

    try:
        # Ensure document_store directory exists
        doc_store = Path(__file__).parent.parent / "data" / "document_store"
        doc_store.mkdir(parents=True, exist_ok=True)

        success_count = 0

        # Try to download real PDFs first
        print("\n📥 Attempting to download PDFs from public sources...")
        print("-" * 70)

        for pdf_info in MEDICAL_PDFS:
            output_path = doc_store / pdf_info["name"]
            if download_pdf(pdf_info["url"], output_path, pdf_info["description"]):
                success_count += 1
            print()

        # Create fallback PDFs
        print("\n📝 Creating fallback medical documents...")
        print("-" * 70)

        fallback_pdfs = []

        # Try importing fpdf first
        try:
            from fpdf import FPDF
            print("✅ fpdf2 available - creating comprehensive PDFs")

            diabetes_pdf = create_simple_diabetes_pdf()
            if diabetes_pdf:
                fallback_pdfs.append(diabetes_pdf)
                print(f"✅ Created: {diabetes_pdf.name}")

            hypertension_pdf = create_simple_hypertension_pdf()
            if hypertension_pdf:
                fallback_pdfs.append(hypertension_pdf)
                print(f"✅ Created: {hypertension_pdf.name}")

        except ImportError:
            print("⚠️  fpdf2 not installed. Skipping fallback PDFs.")
            print("   Install with: pip install fpdf2")

        print("=" * 70)
        print(f"\n✅ Summary:")
        print(f"   Downloaded: {success_count} PDFs from public sources")
        print(f"   Created: {len(fallback_pdfs)} fallback PDFs")
        print(f"   Total: {success_count + len(fallback_pdfs)} documents in {doc_store}")

        if success_count + len(fallback_pdfs) == 0:
            print("\n⚠️  No PDFs were created or downloaded.")
            print("   This may be due to network issues or missing dependencies.")
            return 1

        print("\n📝 Next steps:")
        print("   1. Run ingestion service:")
        print("      python -m services.ingestion.src.main")
        print("   2. Start API server:")
        print("      python -m uvicorn services.api.src.main:app --reload --port 8001")
        print("   3. Use the chat UI:")
        print("      http://localhost:8001/\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
