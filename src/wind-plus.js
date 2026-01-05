import React, { useEffect, useState, useRef } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

// 本地 mock 风场数据（保持不变）
const mockWindData = [
  {
    header: {
      parameterCategory: 2,
      parameterNumber: 2, // U 分量（东西方向）
      lo1: 0,
      la1: 90,
      lo2: 360,
      la2: -90,
      dx: 2.5,
      dy: 2.5,
      nx: 144,
      ny: 73,
      refTime: "2025-01-01 00:00:00",
    },
    data: Array.from({ length: 144 * 73 }, () => (Math.random() - 0.5) * 20),
  },
  {
    header: {
      parameterCategory: 2,
      parameterNumber: 3, // V 分量（南北方向）
      lo1: 0,
      la1: 90,
      lo2: 360,
      la2: -90,
      dx: 2.5,
      dy: 2.5,
      nx: 144,
      ny: 73,
      refTime: "2025-01-01 00:00:00",
    },
    data: Array.from({ length: 144 * 73 }, () => (Math.random() - 0.5) * 20),
  },
];

// 3D 风场粒子渲染核心组件
function CesiumWindLayer({ particleCount, velocityScale, opacity }) {
  const viewerRef = useRef(null); // Cesium 视图实例
  const particlesRef = useRef([]); // 存储所有 3D 粒子

  // 初始化 Cesium 地球（修复核心：异步加载地形）
  useEffect(() => {
    // 配置 Cesium 访问密钥（使用你提供的令牌）
    Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1M2Q2NmQyMC1jMzc4LTRlODQtYWU3YS1kNmI4ZDYzYjNiYTUiLCJpZCI6MzUyMjYzLCJpYXQiOjE3NjA5NjY5ODB9.IVFTW37PlY3gc5cGGjTJBMq9nvFc1A41vxUx5leRT8c";

    // 异步初始化函数（解决地形加载API变更问题）
    const initCesium = async () => {
      // 创建 3D 地球容器
      viewerRef.current = new Cesium.Viewer("cesiumContainer", {
        // 关键修复：使用最新的异步地形加载 API
        terrainProvider: await Cesium.createWorldTerrainAsync(),
        imageryProvider: new Cesium.OpenStreetMapImageryProvider({
          url: "https://a.tile.openstreetmap.org/",
        }),
        baseLayerPicker: false,
        fullscreenButton: false,
        homeButton: false,
      });

      // 设置初始视角
      viewerRef.current.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(110, 20, 500000),
        orientation: {
          pitch: Cesium.Math.toRadians(-30),
          heading: Cesium.Math.toRadians(0),
        },
      });

      // 初始化粒子
      initParticles();
    };

    // 执行初始化
    initCesium();

    // 清理函数
    return () => {
      if (viewerRef.current) {
        viewerRef.current.destroy();
      }
      particlesRef.current = [];
    };
  }, [particleCount, velocityScale, opacity]); // 补充依赖项，修复 ESLint 警告

  // 初始化 3D 粒子（保持逻辑不变）
  const initParticles = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // 清空旧粒子
    particlesRef.current.forEach((primitive) => {
      viewer.scene.primitives.remove(primitive.primitive);
    });
    particlesRef.current = [];

    // 创建新粒子
    for (let i = 0; i < particleCount; i++) {
      const lon = Math.random() * 360 - 180;
      const lat = Math.random() * 180 - 90;
      const height = 1000;

      const { u, v } = getWindDataAt(lon, lat);

      const particle = new Cesium.BillboardCollection({
        billboards: [
          {
            position: Cesium.Cartesian3.fromDegrees(lon, lat, height),
            image: createParticleTexture(opacity),
            scale: 2,
            color: Cesium.Color.CYAN.withAlpha(opacity),
          },
        ],
      });

      particlesRef.current.push({
        primitive: particle,
        lon,
        lat,
        height,
        u,
        v,
      });

      viewer.scene.primitives.add(particle);
    }

    startParticleAnimation();
  };

  // 根据经纬度获取风场数据（修复 ny 未使用的警告）
  const getWindDataAt = (lon, lat) => {
    const uData = mockWindData[0].data;
    const vData = mockWindData[1].data;
    const { nx } = mockWindData[0].header; // 只获取需要的 nx，移除 ny

    // 计算网格索引
    const lonIndex = Math.floor(((lon + 180) % 360) / 2.5);
    const latIndex = Math.floor((90 - lat) / 2.5);
    const dataIndex = latIndex * nx + lonIndex;

    return {
      u: uData[dataIndex] || 0,
      v: vData[dataIndex] || 0,
    };
  };

  // 创建粒子纹理（保持不变）
  const createParticleTexture = (opacity) => {
    const canvas = document.createElement("canvas");
    canvas.width = 10;
    canvas.height = 10;
    const ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.arc(5, 5, 4, 0, 2 * Math.PI);
    ctx.fillStyle = `rgba(0, 255, 255, ${opacity})`;
    ctx.fill();
    return canvas.toDataURL();
  };

  // 粒子动画（保持不变）
  // 粒子动画（修复 undefined 问题）
const startParticleAnimation = () => {
  const viewer = viewerRef.current;
  if (!viewer) return;

  // 先清除旧的事件监听，避免重复绑定导致的累积错误
  const onTickHandler = () => {
    // 遍历粒子前，先过滤掉无效的粒子（未初始化或已被移除的）
    particlesRef.current = particlesRef.current.filter(particle => 
      particle && particle.primitive && particle.primitive.billboards && particle.primitive.billboards.length > 0
    );

    particlesRef.current.forEach((particle) => {
      // 再次检查当前粒子是否有效
      if (!particle || !particle.primitive || !particle.primitive.billboards) return;
      
      const billboard = particle.primitive.billboards[0];
      // 检查广告牌是否存在
      if (!billboard) return;

      // 正常更新逻辑
      const deltaLon = (particle.u * velocityScale) / 100;
      const deltaLat = (particle.v * velocityScale) / 100;
      const newLon = particle.lon + deltaLon;
      const newLat = particle.lat + deltaLat;

      const clampedLon = ((newLon + 180) % 360) - 180;
      const clampedLat = Math.max(-85, Math.min(85, newLat));

      billboard.position = Cesium.Cartesian3.fromDegrees(
        clampedLon,
        clampedLat,
        particle.height
      );

      particle.lon = clampedLon;
      particle.lat = clampedLat;
    });
  };

  // 绑定事件前先移除旧的监听（避免组件更新时重复绑定）
  viewer.clock.onTick.removeEventListener(onTickHandler);
  viewer.clock.onTick.addEventListener(onTickHandler);
};
  return <div id="cesiumContainer" style={{ width: "100%", height: "100%" }} />;
}

// 3D 风场控制面板
function WindDashboard3D({ onBack }) {
  const [velocityScale, setVelocityScale] = useState(0.5);
  const [opacity, setOpacity] = useState(0.7);
  const [particleCount, setParticleCount] = useState(500);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* 顶部导航 */}
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
        <h1 style={{ margin: 0, fontSize: "1.5rem" }}>🌪️ 3D 全球风场分析</h1>
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

      {/* 3D 地球容器 */}
      <div style={{ flex: 1, width: "100%" }}>
        <CesiumWindLayer
          particleCount={particleCount}
          velocityScale={velocityScale}
          opacity={opacity}
        />
      </div>

      {/* 控制面板 */}
      <div
        style={{
          background: "white",
          padding: "1rem",
          borderTop: "1px solid #ddd",
          display: "flex",
          gap: "2rem",
          justifyContent: "center",
          flexWrap: "wrap",
        }}
      >
        <label>
          粒子速度:
          <input
            type="range"
            min="0.1"
            max="2"
            step="0.1"
            value={velocityScale}
            onChange={(e) => setVelocityScale(parseFloat(e.target.value))}
          />{" "}
          {velocityScale.toFixed(1)}
        </label>

        <label>
          粒子透明度:
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
          />{" "}
          {opacity}
        </label>

        <label>
          粒子数量(密度):
          <input
            type="range"
            min="100"
            max="2000"
            step="100"
            value={particleCount}
            onChange={(e) => setParticleCount(parseInt(e.target.value))}
          />{" "}
          {particleCount}
        </label>
      </div>
    </div>
  );
}

export default WindDashboard3D;