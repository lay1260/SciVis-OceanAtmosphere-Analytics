import React, { useEffect, useState, useRef } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-velocity";

// 1. 生成「时间+深度」双维度风场数据
const generateTimeDepthWindData = () => {
  // 配置：3个时间点 + 3个深度层
  const timePoints = [
    "2025-01-01 00:00:00",
    "2025-01-01 12:00:00",
    "2025-01-02 00:00:00"
  ];
  const depthLayers = [
    { id: 0, name: "100m", speedFactor: 1.2 },  // 低空：风速较大（1.2倍基础值）
    { id: 1, name: "500m", speedFactor: 0.8 },  // 中空：风速中等（0.8倍基础值）
    { id: 2, name: "1000m", speedFactor: 0.5 }  // 高空：风速较小（0.5倍基础值）
  ];

  // 生成基础随机风场数据（U/V分量）
  const getBaseWindData = (speedFactor) => {
    return [
      {
        header: {
          parameterCategory: 2,
          parameterNumber: 2, // U分量
          lo1: 0,
          la1: 90,
          lo2: 360,
          la2: -90,
          dx: 2.5,
          dy: 2.5,
          nx: 144,
          ny: 73,
          refTime: "", // 时间后续赋值
          depth: ""    // 深度后续赋值
        },
        data: Array.from({ length: 144 * 73 }, () => (Math.random() - 0.5) * 20 * speedFactor)
      },
      {
        header: {
          parameterCategory: 2,
          parameterNumber: 3, // V分量
          lo1: 0,
          la1: 90,
          lo2: 360,
          la2: -90,
          dx: 2.5,
          dy: 2.5,
          nx: 144,
          ny: 73,
          refTime: "",
          depth: ""
        },
        data: Array.from({ length: 144 * 73 }, () => (Math.random() - 0.5) * 20 * speedFactor)
      }
    ];
  };

  // 组合时间+深度数据：每个时间点对应3个深度层
  return timePoints.map((time) => {
    return {
      time,
      depths: depthLayers.map((depth) => {
        const windData = getBaseWindData(depth.speedFactor);
        // 给每个数据切片添加时间和深度标识
        windData.forEach(item => {
          item.header.refTime = time;
          item.header.depth = depth.name;
        });
        return {
          ...depth,
          windData
        };
      })
    };
  });
};

// 2. 风场图层组件（适配双维度数据）
function WindLayer({ velocityScale, opacity, particleAge, currentWindData }) {
  const map = useMap();
  const velocityLayerRef = useRef(null);

  useEffect(() => {
    // 依赖检查
    if (!window.L || !window.L.velocityLayer) {
      console.error("❌ leaflet-velocity 未加载");
      return;
    }
    if (!currentWindData || currentWindData.length !== 2) {
      console.error("❌ 风场数据格式错误");
      return;
    }

    // 移除旧图层
    if (velocityLayerRef.current && map.hasLayer(velocityLayerRef.current)) {
      map.removeLayer(velocityLayerRef.current);
    }

    // 创建新图层
    try {
      const newLayer = window.L.velocityLayer({
        displayValues: true,
        displayOptions: {
          velocityType: `风速（${currentWindData[0].header.depth}）`, // 显示当前深度
          displayPosition: "bottomleft",
          displayEmptyString: "无风数据",
          angleConvention: "bearingCW",
          speedUnit: "m/s",
        },
        data: currentWindData,
        maxVelocity: 25, // 适配不同深度的风速范围
        velocityScale,
        opacity,
        particleAge,
      });

      velocityLayerRef.current = newLayer;
      newLayer.addTo(map);
      console.log(`✅ 加载成功：${currentWindData[0].header.refTime} · ${currentWindData[0].header.depth}`);

    } catch (err) {
      console.error("❌ 风场图层创建失败：", err.message);
    }

    // 清理函数
    return () => {
      if (velocityLayerRef.current && map.hasLayer(velocityLayerRef.current)) {
        map.removeLayer(velocityLayerRef.current);
      }
    };
  }, [map, velocityScale, opacity, particleAge, currentWindData]);

  return null;
}

// 3. 主组件（时间+深度双滑块）
function WindDashboard({ onBack }) {
  // 基础风场控制
  const [velocityScale, setVelocityScale] = useState(0.005);
  const [opacity, setOpacity] = useState(0.7);
  const [particleAge, setParticleAge] = useState(90);

  // 双维度数据与滑块状态
  const timeDepthWindData = generateTimeDepthWindData(); // 时间-深度数据集合
  const depthLayers = timeDepthWindData[0].depths; // 提取深度配置（所有时间点一致）
  
  // 滑块状态：默认选中第一个时间、第一个深度
  const [currentTimeIndex, setCurrentTimeIndex] = useState(0);
  const [currentDepthIndex, setCurrentDepthIndex] = useState(0);

  // 计算当前要渲染的风场数据（时间+深度匹配）
  const currentTimeData = timeDepthWindData[currentTimeIndex];
  const currentDepthData = currentTimeData.depths[currentDepthIndex];
  const currentWindData = currentDepthData.windData;

  // 滑块事件处理
  const handleTimeChange = (e) => setCurrentTimeIndex(parseInt(e.target.value));
  const handleDepthChange = (e) => setCurrentDepthIndex(parseInt(e.target.value));

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* 顶部导航：显示当前时间+深度 */}
      <div
        style={{
          background: "linear-gradient(135deg, #1e3a8a, #2563eb)",
          color: "white",
          padding: "1rem 2rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "1.5rem" }}>洋流</h1>
          <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.9rem", opacity: 0.9 }}>
            当前：{currentTimeData.time} · 深度 {currentDepthData.name}
          </p>
        </div>
        <button
          onClick={onBack}
          style={{
            background: "rgba(255,255,255,0.2)",
            border: "none",
            color: "white",
            padding: "0.5rem 1rem",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          返回首页
        </button>
      </div>

      {/* 地图 */}
      <MapContainer
        center={[20, 110]}
        zoom={2}
        style={{ flex: 1, width: "100%", minHeight: "400px" }}
        whenCreated={() => console.log("✅ 地图初始化成功")}
      >
         <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | &copy; <a href="https://opentopomap.org/">OpenTopoMap</a>'
          url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png" // 地形图瓦片地址
          maxZoom={14} // 地形图最大缩放级别
          onError={(err) => console.error("❌ 地形图加载失败：", err)}
        />
        <WindLayer
          velocityScale={velocityScale}
          opacity={opacity}
          particleAge={particleAge}
          currentWindData={currentWindData}
        />
      </MapContainer>

      {/* 控制面板：时间滑块 + 深度滑块 + 原有控制项 */}
      <div
        style={{
          background: "white",
          padding: "1rem",
          borderTop: "1px solid #ddd",
          display: "flex",
          flexWrap: "wrap",
          gap: "1.5rem",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        {/* 1. 时间滑块 */}
        <label style={{ display: "flex", alignItems: "center", gap: "0.8rem", minWidth: "280px" }}>
          时间：
          <input
            type="range"
            min="0"
            max={timeDepthWindData.length - 1}
            step="1"
            value={currentTimeIndex}
            onChange={handleTimeChange}
            style={{ width: "120px" }}
          />
          <span style={{ fontSize: "0.85rem", color: "#333" }}>
            {currentTimeData.time.split(" ")[1]}
          </span>
        </label>

        {/* 2. 深度滑块 */}
        <label style={{ display: "flex", alignItems: "center", gap: "0.8rem", minWidth: "250px" }}>
          深度：
          <input
            type="range"
            min="0"
            max={depthLayers.length - 1}
            step="1"
            value={currentDepthIndex}
            onChange={handleDepthChange}
            style={{ width: "120px" }}
          />
          <span style={{ fontSize: "0.85rem", color: "#333" }}>
            {currentDepthData.name}
          </span>
        </label>

        {/* 3. 原有风场控制项 */}
        <label style={{ fontSize: "0.9rem" }}>
          拖尾：
          <input
            type="range"
            min="0.001"
            max="0.02"
            step="0.001"
            value={velocityScale}
            onChange={(e) => setVelocityScale(parseFloat(e.target.value))}
            style={{ width: "80px", margin: "0 0.5rem" }}
          />
          {velocityScale.toFixed(3)}
        </label>

        <label style={{ fontSize: "0.9rem" }}>
          透明度：
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            style={{ width: "80px", margin: "0 0.5rem" }}
          />
          {opacity}
        </label>

        <label style={{ fontSize: "0.9rem" }}>
          寿命：
          <input
            type="range"
            min="10"
            max="200"
            step="10"
            value={particleAge}
            onChange={(e) => setParticleAge(parseInt(e.target.value))}
            style={{ width: "80px", margin: "0 0.5rem" }}
          />
          {particleAge}
        </label>
      </div>
    </div>
  );
}

export default WindDashboard;