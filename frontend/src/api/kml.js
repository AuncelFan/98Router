import { formatHourToHM } from '../utils/format';
import { setDifficulty } from '../ui/difficulty';
import { createMap } from '../ui/map';

export async function parseKmlFile() {
  const fileInput = document.getElementById('kmlFileInput');
  const statusText = document.getElementById('statusText');
  const resultArea = document.getElementById('resultArea');
  if (!fileInput.files?.length) {
    statusText.className = 'error';
    statusText.textContent = '请先选择一个KML文件';
    return;
  }

  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith('.kml')) {
    statusText.className = 'error';
    statusText.textContent = '仅支持 .kml 文件';
    return;
  }

  const formData = new FormData();
  formData.append('kml_file', file);

  try {
    statusText.className = 'loading';
    statusText.textContent = '解析中...';
    resultArea.style.display = 'none';

    const res = await fetch('/api/parse_kml', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(res.status);

    const { data = {} } = await res.json();

    document.getElementById('total_distance_km').textContent =
      (data.total_distance_km || 0).toFixed(1) + ' km';
    document.getElementById('total_elevation_m').textContent =
      Math.round(data.total_elevation_m || 0) + ' m';
    document.getElementById('total_time_h').textContent =
      formatHourToHM(data.total_time_h || 0);
    document.getElementById('speed_mean_km_h').textContent =
      (data.speed_mean_km_h || 0).toFixed(1) + ' km/h';
    document.getElementById('tired_level').textContent =
      (data.tired_level || 1).toFixed(2);
    document.getElementById('elev_level').textContent =
      (data.elev_level || 1).toFixed(2);

    setDifficulty(data.tired_level || 1);

    // 在地图上显示KML轨迹
    createMap(file);

    statusText.className = 'success';
    statusText.textContent = '解析完成';
    resultArea.style.display = 'block';
  } catch (e) {
    statusText.className = 'error';
    statusText.textContent = `解析失败：${e.message}`;
  }
}
