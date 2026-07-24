---
name: Engineering Story
about: Standard template for all engineering work items
title: "[Component or Service] Clear description of the required outcome"
labels: []
---

## Situation

Describe the current problem or opportunity.

* What is happening today?
* Who or what is affected?
* Why does this need to change?
* Include relevant links, screenshots, incidents or supporting evidence.

---

## Task

### User or System Story

**As a** [user, service or system]
**I want** [required capability]
**So that** [expected outcome]

### Scope

**In scope**

* [Requirement]

**Out of scope**

* [Excluded item]

---

## Acceptance Criteria

Use testable Given–When–Then scenarios.

### Primary Scenario

**Given** [starting condition]
**When** [action occurs]
**Then** [expected result]

### Failure or Validation Scenario

**Given** [invalid input, unavailable dependency or restricted access]
**When** [action occurs]
**Then** [safe and expected behaviour]

### Edge Cases

* [Important boundary condition]

---

## Engineering Guardrails

Complete the applicable items before the story is closed.

### Code Quality

* [ ] The change is limited to the agreed scope.
* [ ] Existing behaviour remains backward-compatible unless explicitly approved.
* [ ] Code follows the team's agreed standards and architecture.
* [ ] No temporary code, debug output or unnecessary complexity remains.
* [ ] Error handling is explicit and does not silently ignore failures.

### Test-Driven Development

* [ ] Acceptance criteria were converted into test scenarios.
* [ ] New or changed business logic was developed using Red–Green–Refactor where practical.
* [ ] Bug fixes include a regression test that fails before the fix and passes afterwards.
* [ ] Tests validate behaviour rather than private implementation details.

### Automated Testing

* [ ] Unit tests cover new and changed logic.
* [ ] Positive, negative and important boundary scenarios are covered.
* [ ] Integration or contract tests are updated where service boundaries change.
* [ ] Critical user journeys are covered by Playwright or an equivalent E2E framework.
* [ ] Existing tests pass.
* [ ] No new flaky or skipped tests are introduced without justification.

### Security and Access

* [ ] Authentication and authorisation are enforced at the appropriate layer.
* [ ] Users and services can access only the data and actions they require.
* [ ] Input is validated and sensitive information is not exposed in logs or errors.
* [ ] Secrets and credentials are not stored in source code.
* [ ] Security and dependency scans pass.
* [ ] Tenant, account or customer isolation is tested where applicable.

### Accessibility and User Journey

* [ ] The primary user journey works from start to completion.
* [ ] Loading, empty, success and error states are handled.
* [ ] Keyboard navigation and focus behaviour work correctly.
* [ ] Controls have clear labels and accessible names.
* [ ] The change meets the team's agreed WCAG accessibility level.
* [ ] The experience works across supported devices and browsers.

### Performance and Reliability

* [ ] The change does not introduce unnecessary API calls, database queries or processing.
* [ ] Timeouts, retries and failure behaviour are defined where applicable.
* [ ] Duplicate requests do not create inconsistent data.
* [ ] Performance-sensitive changes are tested against an agreed threshold.
* [ ] Partial failures do not leave the system in an invalid state.

### Observability

* [ ] Important failures are logged with useful diagnostic context.
* [ ] Logs do not contain sensitive information.
* [ ] Metrics or traces are added where the change affects a critical workflow.
* [ ] The team can determine whether the feature is working correctly after deployment.

### Documentation

* [ ] Relevant README, API or technical documentation is updated.
* [ ] Configuration or environment changes are documented.
* [ ] Deployment, migration or rollback steps are documented where required.
* [ ] Known limitations are recorded.

---

## Result

Define what success looks like.

* Expected user or system outcome:
* Expected business or operational outcome:
* Success metric or validation method:
* Monitoring period, where applicable:

---

## Pull Request Evidence

The pull request should include:

* [ ] Link to this story.
* [ ] Summary of the change.
* [ ] Test evidence.
* [ ] Screenshots or recordings for UI changes.
* [ ] Security, accessibility or performance evidence where relevant.
* [ ] Deployment and rollback notes for higher-risk changes.
* [ ] Known risks or follow-up work.

---

## Definition of Done

The story is complete when:

* [ ] All acceptance criteria pass.
* [ ] Required tests pass locally and in CI.
* [ ] No critical or high-severity security or quality issues remain.
* [ ] The critical user or system journey has been validated.
* [ ] Documentation is updated.
* [ ] Code review is complete.
* [ ] The change has been deployed or is ready for deployment through the standard pipeline.
* [ ] Post-deployment validation is defined or completed.
* [ ] Any exception is documented with an owner and reason.

---

## Exceptions

For any guardrail marked not applicable, record:

**Guardrail:**
**Reason:**
**Alternative validation:**
**Residual risk:**
**Approved by:**
