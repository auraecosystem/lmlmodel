{
  "protocol": "GOD/0.1",
  "kind": "coordinator",
  "name": "Script.god",
  "version": "0.1.0",
  "description": "Standalone root coordinator for independent LMLM scripts and model participants.",
  "listen": "127.0.0.1:8787",
  "scripts": [
    {"name": "Script.dm", "role": "direct-messaging", "manifest": "scripts/Script.dm"},
    {"name": "Script.byod", "role": "data-ingestion", "manifest": "scripts/Script.byod"},
    {"name": "Script.sync", "role": "knowledge-synchronization", "manifest": "scripts/Script.sync"},
    {"name": "Script.policy", "role": "policy-enforcement", "manifest": "scripts/Script.policy"},
    {"name": "Script.verify", "role": "verification", "manifest": "scripts/Script.verify"}
  ],
  "security": {
    "default_model_access": "deny",
    "require_registered_recipient": true,
    "require_ack_for_tasks": true,
    "allow_arbitrary_url_fetch": false,
    "allow_model_to_override_policy": false,
    "persistent_memory_requires_verification": true
  },
  "limits": {
    "max_message_bytes": 1048576,
    "request_timeout_seconds": 30,
    "max_context_bytes": 262144
  }
}
