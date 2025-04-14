def get_prompt_template(level):
    templates = {
        1: "Give a general hint that nudges the student in the right direction.",
        2: "Give a conceptual hint or ask a clarifying question that leads to the answer.",
        3: "Provide a more detailed guided explanation without giving the full answer."
    }
    return templates.get(level, templates[2])
