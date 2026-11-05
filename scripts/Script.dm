{
  "protocol": "GOD/0.1",
  "kind": "script",
  "name": "Script.dm",
  "version": "0.1.0",
  "role": "direct-messaging",
  "accepts": ["TASK", "CONTEXT", "CANCEL"],
  "emits": ["ACK", "PROGRESS", "RESULT", "ERROR", "BLOCKED"],
  "operations": {
    "DM": "Deliver a task or context to one registered recipient.",
    "SHARE": "Share a named context object with an explicitly selected recipient."
  },
  "rules": {
    "recipient_must_be_registered": true,
    "preserve_message_id": true,
    "propagate_correlation_id": true,
    "never_escalate_recipient_permissions": true
  }
}
