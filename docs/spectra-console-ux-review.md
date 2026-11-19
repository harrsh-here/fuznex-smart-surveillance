# SPECTRA AI Diagnostic Console — UX review

Reviewed 2026-11-19 against practical patterns from Nielsen Norman Group,
Microsoft Fluent 2, and Grafana.

## Findings in the 1259f33 console

1. **The live screen asks for expert choices before the first task.** Camera
   source, three detection switches, private recognition, Stop, and Start are
   all placed at the same visual level. The empty preview tells the user to
   configure models in another tab, but offers no route there.
2. **Navigation describes implementation, not user goals.** Five horizontal
   tabs compete for attention. `Diagnostics` overlaps with `Logs & sessions`;
   `People` and `Enroll person` split one simple job over two places.
3. **The hierarchy is visually flat.** The default-looking controls, dense
   toolbar, large empty panels, and equal treatment of primary/destructive and
   secondary actions make scanning and confidence difficult.
4. **The state feedback is easy to miss and sometimes misleading.** A global
   status line can show an unrelated enrollment validation error on the Live
   page. There is no concise “what should I do next?” state or clear model
   readiness indicator.
5. **Logs and exports require operational knowledge.** Ground-truth flags and
   the test session setup appear before a user can understand the normal
   workflow. The raw live text log has no severity, search, or simple
   explanation of what is saved.

## Design decisions

The refactor uses four task-oriented destinations: **Live test**, **People**,
**Diagnostics**, and **Export results**. This follows Fluent’s guidance to
keep navigation short, clear, scannable, and goal-oriented, and to show a
strong active location. Advanced setup is a one-time dialog launched only when
needed. The active page owns its actions and status.

The live page uses one clear primary action, a camera-forward surface, a small
health summary, and a compact activity feed. This applies NN/g’s progressive
disclosure approach rather than presenting the whole application at once.
Diagnostics and logs adopt Grafana’s queryless-drilldown idea: a visible health
summary first, with filterable, severity-labelled events and optional detail
rather than configuration language up front.

## Acceptance path

1. Launch → choose **Start camera**.
2. If setup is incomplete, choose model files once in **Set up models** and
   return automatically to Live test.
3. Select **People** → enter a display name, confirm consent, then choose
   **Start guided enrollment** while the camera is running.
4. Select **Diagnostics** to see readiness, performance, and errors.
5. Select **Export results** → choose **Start a saved test session**, then
   return to Live test. No terminal commands or model paths are required in
   the everyday path.
