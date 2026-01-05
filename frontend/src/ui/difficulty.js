const MIN_LEVEL = 1;
const MAX_LEVEL = 10;
const ROW_RANGE = 0.8;

let container, arrow, valueText, scaleBox;

export function initDifficulty() {
  container = document.querySelector('.difficulty-container');
  arrow = document.getElementById('diffArrow');
  valueText = document.getElementById('diffValue');
  scaleBox = document.getElementById('diffScale');

  createScale();
}

function createScale() {
  scaleBox.innerHTML = Array.from(
    { length: MAX_LEVEL },
    (_, i) => `<span class="scale-num">${i + 1}</span>`
  ).join('');
}

export function setDifficulty(value) {
  const safe = Math.max(MIN_LEVEL, Math.min(MAX_LEVEL, value));
  const percent =
    ((safe - MIN_LEVEL) / (MAX_LEVEL - MIN_LEVEL)) * 100 * ROW_RANGE +
    (100 * (1 - ROW_RANGE)) / 2;

  arrow.style.left = `${percent}%`;
  valueText.style.left = `${percent}%`;
  valueText.textContent = safe.toFixed(2);
  container.style.display = 'block';
}
