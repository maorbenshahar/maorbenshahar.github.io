// Registers a Lean grammar for highlight.js (which ships none) and runs
// highlighting. Must load AFTER the highlight.js core script.
hljs.registerLanguage('lean', function (hljs) {
  const KEYWORDS = {
    keyword:
      'theorem lemma def example axiom inductive structure class instance ' +
      'fun match with do let mut if then else by from show have suffices ' +
      'calc forall exists open import namespace end section variable variables ' +
      'where deriving extends abbrev notation macro syntax universe',
    built_in: 'Prop Type Sort Nat Int Bool String List Array Option',
    literal: 'true false'
  };
  return {
    name: 'Lean',
    keywords: KEYWORDS,
    contains: [
      hljs.COMMENT('--', '$'),
      hljs.COMMENT('/-', '-/'),
      { className: 'string', begin: '"', end: '"' },
      { className: 'symbol', begin: /:=|=>|->|→|∀|∃|λ|↦|∧|∨|¬|≤|≥|≠/ },
      { className: 'number', begin: /\b\d+\b/ }
    ]
  };
});
document.addEventListener('DOMContentLoaded', function () {
  // Strip leading/trailing blank lines so the HTML source can be formatted
  // nicely (code on its own line after <code>) without rendering a gap.
  // Only blank lines are removed; indentation inside the block is preserved.
  document.querySelectorAll('pre code').forEach(function (el) {
    el.textContent = el.textContent.replace(/^\n+/, '').replace(/\s+$/, '');
  });
  hljs.highlightAll();  // block code: <pre><code class="language-lean">
  // Inline code: <code class="language-lean">...</code> not inside a <pre>.
  document.querySelectorAll('code.language-lean').forEach(function (el) {
    if (!el.closest('pre')) hljs.highlightElement(el);
  });
});
