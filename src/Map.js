import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, ImageOverlay, useMap } from "react-leaflet";
import L from 'leaflet'; // 引入Leaflet核心库
import "leaflet/dist/leaflet.css";

// 温度数据的精确地理边界配置（根据实际数据调整）
const TEMP_DATA_GEO_BOUNDS = {
  topLeft: [60, -180],    // [纬度, 经度] - 温度数据左上角对应的地理坐标
  bottomRight: [-60, 180] // [纬度, 经度] - 温度数据右下角对应的地理坐标
};

// 温度图层组件 - 修复边界计算错误
function TemperatureLayer({ imgUrl, showTemperature }) {
  const map = useMap();
  
  // 计算温度数据的地理边界（使用Leaflet的L.latLngBounds）
  const temperatureBounds = L.latLngBounds(
    TEMP_DATA_GEO_BOUNDS.bottomRight,
    TEMP_DATA_GEO_BOUNDS.topLeft
  );

  if (!showTemperature || !imgUrl) return null;

  return (
    <ImageOverlay 
      url={imgUrl}
      bounds={temperatureBounds}  // 使用Leaflet的Bounds对象
      opacity={0.85}
      interactive={false}
      className="temperature-overlay"
    />
  );
}

function Map({ onBack }) {
  const [imgUrl, setImgUrl] = useState(null);
  const [showTemperature, setShowTemperature] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      try {
        const res = await fetch("/2d_ndarray_data.txt");
        const text = await res.text();
        const rows = text.trim().split("\n");
        const data = rows.map((r) =>
          r.split(/\s+/).map((v) => (v === "nan" ? NaN : parseFloat(v)))
        );

        const height = data.length;
        const width = data[0]?.length || 0;
        
        if (width === 0 || height === 0) {
          throw new Error("Invalid temperature data format");
        }

        // 计算地理边界宽高比
        const geoWidth = TEMP_DATA_GEO_BOUNDS.bottomRight[1] - TEMP_DATA_GEO_BOUNDS.topLeft[1];
        const geoHeight = TEMP_DATA_GEO_BOUNDS.topLeft[0] - TEMP_DATA_GEO_BOUNDS.bottomRight[0];
        const dataAspectRatio = width / height;

        // 创建与地理比例匹配的画布
        let targetWidth, targetHeight;
        if (dataAspectRatio > (geoWidth / geoHeight)) {
          targetWidth = 1440;
          targetHeight = Math.round(targetWidth / dataAspectRatio);
        } else {
          targetHeight = 800;
          targetWidth = Math.round(targetHeight * dataAspectRatio);
        }

        const canvas = document.createElement("canvas");
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(targetWidth, targetHeight);

        const vmin = 250, vmax = 310;

        function turboColormap(t) {
          const colors = [
            [48, 18, 59], [65, 69, 171], [44, 146, 240],
            [41, 213, 156], [220, 226, 31], [252, 165, 10], [252, 93, 7]
          ];

          const n = colors.length - 1;
          const idx = t * n;
          const i = Math.floor(idx);
          const f = idx - i;

          if (i >= n) return colors[n];
          const c0 = colors[i];
          const c1 = colors[i + 1];

          return [
            Math.round(c0[0] + (c1[0] - c0[0]) * f),
            Math.round(c0[1] + (c1[1] - c0[1]) * f),
            Math.round(c0[2] + (c1[2] - c0[2]) * f)
          ];
        }

        function getColor(value) {
          if (isNaN(value)) return [200, 200, 200];
          const t = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)));
          return turboColormap(t);
        }

        // 按地理坐标比例采样
        for (let y = 0; y < targetHeight; y++) {
          const dataY = Math.floor((y / targetHeight) * height);
          for (let x = 0; x < targetWidth; x++) {
            const dataX = Math.floor((x / targetWidth) * width);
            const safeX = Math.min(width - 1, Math.max(0, dataX));
            const safeY = Math.min(height - 1, Math.max(0, dataY));
            
            const value = data[safeY][safeX];
            const [r, g, b] = getColor(value);
            const pixelIndex = (y * targetWidth + x) * 4;
            
            imgData.data[pixelIndex] = r;
            imgData.data[pixelIndex + 1] = g;
            imgData.data[pixelIndex + 2] = b;
            imgData.data[pixelIndex + 3] = 255;
          }
        }

        ctx.putImageData(imgData, 0, 0);
        setImgUrl(canvas.toDataURL());
      } catch (error) {
        console.error("Error loading temperature data:", error);
      } finally {
        setIsLoading(false);
      }
    }

    loadData();
  }, []);

  // 添加CSS确保缩放质量
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .temperature-overlay {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const buttonStyle = {
    padding: "10px 16px",
    fontSize: "14px",
    borderRadius: "5px",
    cursor: "pointer",
    zIndex: 1000,
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    height: "42px",
    boxSizing: "border-box",
  };

  return (
    <div style={{ height: "100vh", position: "relative" }}>
      <MapContainer
        center={[20, 110]}
        zoom={2}
        style={{ width: "100%", height: "100%" }}
        maxZoom={10}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        />
        
        <TemperatureLayer 
          imgUrl={imgUrl} 
          showTemperature={showTemperature}
        />
      </MapContainer>

      <div style={{
        position: "absolute",
        top: "20px",
        left: "20px",
        display: "flex",
        gap: "10px",
        alignItems: "center",
      }}>
        <button
          onClick={onBack}
          style={{
            ...buttonStyle,
            background: "#fff",
            border: "1px solid #ccc",
            width: "100px",
          }}
        >
          ← 返回首页
        </button>

        <button
          onClick={() => setShowTemperature(!showTemperature)}
          disabled={isLoading}
          style={{
            ...buttonStyle,
            background: showTemperature ? "#fc5d07" : "#4145ab",
            color: "#fff",
            border: "none",
            width: "130px",
            opacity: isLoading ? 0.7 : 1,
          }}
        >
          {isLoading ? "加载中..." : showTemperature ? "隐藏温度图" : "显示温度图"}
        </button>
      </div>

      {isLoading && (
        <div style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          background: "rgba(255,255,255,0.9)",
          padding: "20px 30px",
          borderRadius: "8px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
          zIndex: 1000
        }}>
          正在加载温度数据...
        </div>
      )}

      {showTemperature && !isLoading && (
        <div
          style={{
            position: "absolute",
            bottom: "20px",
            left: "20px",
            background: "rgba(255,255,255,0.9)",
            padding: "12px 15px",
            borderRadius: "6px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
            zIndex: 1000,
            width: "280px",
          }}
        >
          <div style={{ fontSize: "14px", marginBottom: "8px", fontWeight: 500 }}>
            🌡️ Temperature (K)
          </div>
          <div
            style={{
              background: "linear-gradient(to right, #30123b, #4145ab, #2c92f0, #29d59c, #dce21f, #fca50a, #fc5d07)",
              width: "100%",
              height: "15px",
              borderRadius: "4px",
              marginBottom: "6px",
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#555" }}>
            <span>250K</span>
            <span>280K</span>
            <span>310K</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default Map;
    