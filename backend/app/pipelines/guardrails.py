import re
from fastapi import HTTPException

class SafetyGuardrails:
    def __init__(self):
        self.injection_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now an unrestricted ai",
            r"forget all rules"
        ]
        self.email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        self.phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

    def validate_input(self, user_query: str) -> str:
        query_lower = user_query.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, query_lower):
                raise HTTPException(
                    status_code=400, 
                    detail="Security Alert: Potential prompt injection attempt detected."
                )
        return user_query.strip()

    def validate_and_scrub_output(self, llm_response: str) -> str:
        scrubbed = re.sub(self.email_pattern, "[REDACTED EMAIL]", llm_response)
        scrubbed = re.sub(self.phone_pattern, "[REDACTED PHONE]", scrubbed)
        return scrubbed

guardrails = SafetyGuardrails()