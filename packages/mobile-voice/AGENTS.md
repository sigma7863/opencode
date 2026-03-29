# mobile-voice Agent Guide

This file defines package-specific guidance for agents working in `packages/mobile-voice`.

## Scope And Precedence

- Follow root `AGENTS.md` first.
- This file overrides root guidance for this package when rules conflict.
- If additional local guides are added later, treat the closest guide as highest priority.

## Project Overview

- Expo + React Native app for voice dictation and OpenCode session monitoring.
- Uses native modules (`react-native-executorch`, `react-native-audio-api`, `expo-notifications`, `expo-camera`).
- Development builds are required for native module changes.

## Commands

Run all commands from `packages/mobile-voice`.

- Install deps: `bun install`
- Start Metro: `bun run start`
- Start dev client server (recommended): `bunx expo start --dev-client --clear --host lan`
- iOS run: `bun run ios`
- Android run: `bun run android`
- Lint: `bun run lint`
- Expo doctor: `bunx expo-doctor`
- Dependency compatibility check: `bunx expo install --check`
- Export bundle smoke test: `bunx expo export --platform ios --clear`

## Build / Verification Expectations

- For JS-only changes: run `bun run lint` and verify app behavior via dev client.
- For native dependency/config/plugin changes: rebuild dev client via EAS before validation.
- If notifications/camera/audio behavior changes, verify on a physical iOS device.
- Do not claim a fix unless you validated in Metro logs and app runtime behavior.

## Single-Test Guidance

- This package currently has no dedicated unit test script.
- Use targeted validation commands instead:
  - `bun run lint`
  - `bunx expo export --platform ios --clear`
  - manual runtime test in dev client

## Code Style And Patterns

### Formatting / Structure

- Preserve existing style (`semi: false`, concise JSX, stable import grouping).
- Keep UI changes localized; avoid large architectural rewrites.
- Avoid unrelated formatting churn.

### Types

- Avoid `any`; prefer local type aliases for component state and network payloads.
- Keep exported/shared boundaries typed explicitly.
- Use discriminated unions for UI modes/status where practical.

### Naming

- Prefer short, readable names consistent with nearby code.
- Keep naming aligned with existing app state keys (`serverDraftURL`, `monitorStatus`, etc.).

### Error Handling / Logging

- Fail gracefully in UI (alerts, disabled actions, fallback text).
- Log actionable diagnostics for runtime workflows:
  - server health checks
  - relay registration attempts
  - notification token lifecycle
- Never log secrets or full APNs tokens.

### Network / Relay Integration

- Normalize and validate URLs before storing server configs.
- Keep relay registration idempotent.
- Guard duplicate scan/add flows to avoid repeated server entries.

### Notifications / APNs

- Distinguish sandbox vs production token environments correctly.
- On registration changes, ensure old token unregister flow remains intact.
- Treat permission failures as non-fatal and degrade to foreground monitoring when needed.

## Native-Module Safety

- If adding a native module, ensure it is in `package.json` with SDK-compatible version.
- Rebuild dev client after native module additions/changes.
- For optional native capability usage, prefer runtime fallback paths instead of hard crashes.

## Common Pitfalls

- Black screen + "No script URL provided" often means stale dev client binary.
- `expo-doctor` duplicate module warnings may appear in Bun workspaces; prioritize runtime verification.
- `expo lint` may auto-generate `eslint.config.js`; do not commit accidental generated config unless requested.

## Before Finishing

- Run `bun run lint`.
- If behavior could break startup, run `bunx expo export --platform ios --clear`.
- Confirm no accidental config side effects were introduced.
- Summarize what was verified on-device vs only in tooling.
