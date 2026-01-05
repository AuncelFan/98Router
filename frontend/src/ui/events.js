import { parseKmlFile } from '../api/kml';
import { exportResultImage } from '../export/image';

export function bindEvents() {
  // 解析按钮
  document.getElementById('parseBtn')?.addEventListener('click', parseKmlFile);

  // 文件选择按钮
  document.querySelector('.custom-file-btn')?.addEventListener('click', () => {
    document.getElementById('kmlFileInput').click();
  });

  // 显示选择的文件名
  document.getElementById('kmlFileInput')?.addEventListener('change', e => {
    const text = document.getElementById('fileNameText');
    const file = e.target.files?.[0];
    text.textContent = file ? file.name : '未选择文件';
  });

  // 显示/关闭时间信息按钮
  document.getElementById('show-speed-btn')?.addEventListener('click', () => {
    const timeInfo = document.getElementById('time-info');
    if (timeInfo.style.display === 'none') {
      timeInfo.style.display = 'flex';
    } else {
      timeInfo.style.display = 'none';
    }
  });

  // 保存结果按钮
  document.getElementById('save-result-btn')?.addEventListener('click', () => {
    exportResultImage('result-info');
  });
}
