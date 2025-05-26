![System](https://github.iu.edu/hagupta/TA-Lite/blob/main/SystemArchitecture.png?raw=true)

### Feature	Adds	Time
- 👨‍🏫 Teacher dashboard (show what students are asking)	Insight for professors	1 day
- 🧠 Adaptive hinting (more hints on request)	Feels more like a human TA	1 day
- 📝 Logs of student questions + feedback	Research or eval data	Half day
- 💬 Few-shot hint styles (Socratic, conceptual, etc.)	More controlled output	1–2 hrs


## Teacher Portal
### Feature	Description
- Set Constraints / Hint Levels	Define depth of hints (Level 1–3) per topic or question type
- Edit Prompt Templates	Modify the LLM prompt for hinting or explanation style
- Upload Course Materials	PDFs, notes, textbook sections — stored for retrieval
- Add Reference Links	Curated external links or sources for students to use
- View Student Logs	See student questions, hint usage, topic trends

## Student Portal
### Feature	Description
- Query Resolution	Ask a question and get guided hints based on teacher rules
- Lecture Summarizer	Upload or select a lecture → get summary + key takeaways
- Notes Maker	Generate crisp notes from course files or custom uploads
- Reference Materials	View links + files uploaded by the teacher

```
llm_doubt_solver/
├── student_portal/
│   └── app.py         # Streamlit interface for students
├── teacher_portal/
│   └── app.py         # Streamlit interface for teachers
├── backend/
│   ├── rag_engine.py  # Retrieval logic
│   ├── prompt_logic.py# Dynamic prompt construction
│   └── utils.py       # PDF parsing, text splitting, etc.
├── config/
│   └── prompt_settings.json   # Editable by teacher
│   └── constraints.json       # Hint depth, allowed topics etc.
├── data/
│   ├── course_materials/      # PDF/Text uploads
│   ├── reference_links.json   # External links
│   └── student_logs.json      # Optional
```
