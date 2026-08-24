mkdir -p .github && cat << 'EOF' > .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "Africa/Lagos"
    open-pull-requests-limit: 5
    labels:
      - "ci"
      - "dependencies"
    commit-message:
      prefix: "ci(deps)"
    groups:
      actions-updates:
        patterns:
          - "*"

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "Africa/Lagos"
    open-pull-requests-limit: 5
    labels:
      - "python"
      - "dependencies"
    commit-message:
      prefix: "build(deps-python)"
    groups:
      minor-and-patch:
        update-types:
          - "minor"
          - "patch"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    labels:
      - "docker"
      - "dependencies"
    commit-message:
      prefix: "build(deps-docker)"
EOF
