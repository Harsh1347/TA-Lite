-- Create the student_queries table
CREATE TABLE IF NOT EXISTS student_queries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT NOT NULL,
    question TEXT NOT NULL,
    llm_answer TEXT,
    topic TEXT,
    summary TEXT,
    difficulty INT CHECK (difficulty >= 1 AND difficulty <= 5),
    needs_review BOOLEAN DEFAULT FALSE,
    key_concepts TEXT,
    prerequisites TEXT,
    learning_objectives TEXT,
    source_material TEXT,
    timestamp TIMESTAMPTZ DEFAULT now(),
    has_mcq BOOLEAN DEFAULT FALSE,
    has_flashcards BOOLEAN DEFAULT FALSE,
    mcq_generated_at TIMESTAMPTZ,
    flashcards_generated_at TIMESTAMPTZ,
    engagement_score INT DEFAULT 0,
    feedback_rating INT CHECK (feedback_rating >= 1 AND feedback_rating <= 5),
    CONSTRAINT valid_difficulty CHECK (difficulty >= 1 AND difficulty <= 5)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_student_queries_student_id ON student_queries(student_id);
CREATE INDEX IF NOT EXISTS idx_student_queries_topic ON student_queries(topic);
CREATE INDEX IF NOT EXISTS idx_student_queries_timestamp ON student_queries(timestamp); 