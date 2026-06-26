// MathJax config. Must load BEFORE the MathJax script itself.
// Inline: $...$ or \(...\)   Display: $$...$$ or \[...\]
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
};
