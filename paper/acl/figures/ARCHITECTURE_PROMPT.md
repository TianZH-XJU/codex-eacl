# Architecture figure prompt

The paper uses `figures/fig_pipeline.pdf`, a deterministic TikZ diagram built in
`paper/acl/sec_method.tex`. It is editable, vector, and prints cleanly at column
width. The AI-image route was attempted through the OpenCode image tool, but the
provider returned `Images API is not supported for this platform`, so no
generated image is used in the paper.

If an AI-rendered conceptual illustration is wanted later, use the prompt below.
Keep the TikZ version as the publication figure: generated images tend to garble
short labels and cannot be edited reliably.

---

Publication-quality flat vector-style architecture diagram for an ACL paper,
white background, wide 16:9 composition, clean sans-serif labels. Left to right:
rounded rectangle labeled "input x" -> rounded rectangle "frozen model M" ->
rounded rectangle "NOUL call p(x)" -> rounded rectangle "band gate l <= p <= h".
From the gate, two arrows: one up to "CHOICE call / second view", one down to
"NOUL label / no second call". Below, a separate dashed offline row:
"immutable cache" -> "fitness J(g)" -> "GA / ES / NSGA-II". Muted blue and grey
palette, thin dark strokes, no shadows, no 3D, no decorative icons, generous
spacing, text large and legible at 300 dpi. Do not add any other text or numbers.

---

A more conceptual alternative, if the goal is a teaser rather than a pipeline
diagram:

Minimal editorial illustration for a systems paper: a frozen language model
shown as a single grey block, a typed question entering from the left, a
probability dial on its output, and a two-branch switch after the dial. One
branch is a cheap binary label, the other a second question to the same frozen
block. A dashed loop below the switch represents an offline cache replaying
candidate gates. Off-white background, muted slate and amber palette, thin
lines, flat vector look, no text except the labels "input", "frozen model",
"probability", "second call", and "cache". Wide composition, generous negative
space, no 3D, no glow, no stock-photo elements.
