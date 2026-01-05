import './style.css';
import { initDifficulty } from './ui/difficulty';
import { bindEvents } from './ui/events';

document.addEventListener('DOMContentLoaded', () => {
  initDifficulty();
  bindEvents();
});