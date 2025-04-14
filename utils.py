from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def get_prompt_template(level):
    templates = {
        1: "Give a general hint that nudges the student in the right direction.",
        2: "Give a conceptual hint or ask a clarifying question that leads to the answer.",
        3: "Provide a more detailed guided explanation without giving the full answer."
    }
    return templates.get(level, templates[2])
