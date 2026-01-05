import 'ol/ol.css';
import { Map, View } from 'ol'
import XYZ from 'ol/source/XYZ';
import { fromLonLat } from 'ol/proj';
import TileLayer from 'ol/layer/Tile';
import VectorLayer from 'ol/layer/Vector';
import KML from 'ol/format/KML.js';
import VectorSource from 'ol/source/Vector.js';

export function createMap(file) {
  // 检查是否已经初始化地图
  if (window.mapInstance) {
    // 先清理旧的地图实例
    window.mapInstance.dispose();
  }

  // 高德普通地图瓦片
  const gaodeLayer = new TileLayer({
    source: new XYZ({
      url: 'http://wprd0{1-4}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}',
      subdomains: ['1', '2', '3', '4'],
      maxZoom: 16,
      projection: 'EPSG:3857'
    })
  });

  // KML轨迹图层
  const tempUrl = URL.createObjectURL(file);
  const vectorSource = new VectorSource({
    url: tempUrl,
    format: new KML({ extractStyles: false }),
  });
  const vector = new VectorLayer({
    source: vectorSource,
    style: {
      'stroke-color': '#FF0000',
      'stroke-width': 3
    }
  });

  // 初始化地图
  const mapDom = document.getElementById('map');
  const map = new Map({
    target: mapDom,
    layers: [gaodeLayer, vector],
    view: new View({
      center: fromLonLat([121.4737, 31.2304]),
      zoom: 10
    })
  });

  // 轨迹加载后自动缩放到合适视野
  vectorSource.on('featuresloadend', function () {
    const vectorExtent = vectorSource.getExtent();
    map.getView().fit(vectorExtent, { padding: [50, 50, 50, 50], maxZoom: 15 });
  });

  // 将地图实例挂载到全局并显示
  window.mapInstance = map;
  mapDom.style.display = 'block';
}