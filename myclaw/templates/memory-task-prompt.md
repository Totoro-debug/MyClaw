Maintain the MyClaw Long-term Memory from new Conversation Summaries.
Use read_file to inspect exactly {long_term_path}.
Use edit_file only when stable information should be retained, and edit exactly that file.
Keep the four sections: User Info, User Preference, Project Fact, and Lesson.
Do not store transient activity, raw summaries, or duplicate facts.
If no durable update is needed, do not call edit_file.
