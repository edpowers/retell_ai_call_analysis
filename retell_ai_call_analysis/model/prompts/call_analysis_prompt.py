import json
from typing import Any


class CallAnalysisPrompt:
    def create_system_prompt(self) -> str:
        return """
        You are an expert call analyzer. Analyze the call transcript and identify issues based on the criteria provided.
        """

    def create_analysis_prompt(
        self,
        transcript: str,
        dynamic_vars: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> str:
        """
        Create a prompt for OpenAI to analyze the call.

        Args:
            transcript: The call transcript
            dynamic_vars: Dictionary of dynamic variables
            tool_calls: List of tool call dictionaries

        Returns:
            A formatted prompt string for OpenAI
        """
        prompt = """
Please analyze this call transcript and identify any issues based on the following criteria:

1. Dynamic Variable Mismatch: Check if the details in retell_llm_dynamic_variables match what's mentioned by the user in the transcript. Only flag this if the user explicitly provided information that contradicted the dynamic variables, not if the user didn't provide information at all. Ignore any information provided by the agent.
2. Early Hang-up: Determine if the user hung up before the agent could complete their task.
3. Contact Information Issues: Check if the agent had trouble understanding the user's contact information.
4. AI Detection: Check if the user asked if the agent was an AI/robot.
5. Booking Issues: Determine if the call was nearly successful but the agent wasn't able to book properly.
6. Wrong Business Type: Check if the call was made to a non-dental business (e.g., law office, restaurant, etc.). We should only be calling dental clinics.
7. Doctor Not Found: Check if the call was made to a dental clinic but the specific doctor mentioned in the dynamic variables doesn't work there.
8. Booking Incomplete: If the user agrees to a booking but there's no confirmation that the booking was successful, set this to true.

Dynamic Variables:
{dynamic_vars}


Tool Calls:
{tool_calls}


Transcript:
{transcript}

Provide your analysis in JSON format with the following fields:
- issue_type: The primary type of issue (one of: "dynamic_var_mismatch", "early_hang_up", "contact_info_issue", "ai_detection", "booking_issue", "wrong_business_type", "doctor_not_found", "booking_incomplete", "other", or null if successful)
- issue_description: A brief description of the specific issue
- success_status: "success", "failed", or "partial"
- contact_info_captured: boolean indicating if contact information was successfully captured
- booking_attempted: boolean indicating if booking was attempted
- booking_successful: boolean indicating if booking was successful
- ai_detection_question: boolean indicating if user asked if agent was AI
- hang_up_early: boolean indicating if user hung up early
- dynamic_var_mismatch: boolean indicating if there was a mismatch between dynamic variables and user-provided information
- needs_human_review: boolean indicating if this call requires manual review by a human
- has_issue: boolean indicating if any issues were detected in the call
- notes: Any additional observations

For the issue_type field:
- Use "wrong_business_type" ONLY if there's explicit evidence that the business is not a dental clinic (e.g., they clearly state they are a law office, restaurant, etc.)
- IMPORTANT: Assume all businesses are dental clinics unless explicitly stated otherwise. Business names like "GoDent" or similar should be assumed to be dental clinics.
- Use "doctor_not_found" if the business is a dental clinic but the specific doctor mentioned in the dynamic variables doesn't work there
- Both "wrong_business_type" and "doctor_not_found" should be considered serious errors and should always set has_issue to true and needs_human_review to true

For the dynamic_var_mismatch field:
- Set to true ONLY if the user explicitly provided information that contradicted the dynamic variables
- Set to false if the user didn't provide information or if the information matches the dynamic variables
- IMPORTANT: Only consider information provided by the user, not information stated by the agent

For the needs_human_review field, set it to true ONLY if:
- There are complex issues that AI might not fully understand
- The call outcome is ambiguous and requires human judgment
- There are potential technical problems that need human verification
- The booking process had errors but contact information was captured and there's a chance for follow-up
- There's evidence of a system malfunction or unexpected behavior
- The call was made to a wrong business type (non-dental business)
- The doctor mentioned in the dynamic variables doesn't work at the dental clinic

Do NOT set needs_human_review to true for:
- Simple failed calls where the user didn't provide information
- Voicemails or messages left with no user interaction
- Calls that went to voicemail, even if there's uncertainty about whether the doctor works at the clinic
- Routine hang-ups or disconnections
- Cases where the outcome is clear (either clearly successful or clearly failed)

For the has_issue field, set it to true if any issues were detected that affected the call outcome.
"""
        return prompt.format(
            transcript=transcript,
            dynamic_vars=json.dumps(dynamic_vars, indent=2),
            tool_calls=json.dumps(tool_calls, indent=2),
        )
