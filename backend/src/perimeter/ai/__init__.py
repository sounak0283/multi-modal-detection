"""AI video analysis and chat over Amazon Bedrock (Phase J).

Everything here reads alerts and clips the rest of the system already produces; nothing in
detection, alerting or email depends on it, and a Bedrock outage must never affect them.
"""
