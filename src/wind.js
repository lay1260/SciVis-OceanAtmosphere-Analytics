import React, { useEffect, useRef, useState, useCallback } from 'react';

// 高度轴组件
export const HeightAxis = ({ currentHeight, onHeightChange }) => {
  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      left: '20px',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderRadius: '8px',
      padding: '12px 16px',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.15)',
      zIndex: 2001,
      display: 'flex',
      gap: '12px',
      alignItems: 'center'
    }}>
      <span style={{ fontSize: '14px', fontWeight: 600, color: '#2d3748', marginRight: '8px' }}>高度轴：</span>
      <button
        onClick={() => onHeightChange(1)}
        style={{
          backgroundColor: currentHeight === 1 ? '#10b981' : '#e2e8f0',
          color: currentHeight === 1 ? 'white' : '#4a5568',
          border: 'none',
          borderRadius: '6px',
          padding: '8px 16px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}
        onMouseOver={(e) => {
          if (currentHeight !== 1) {
            e.target.style.backgroundColor = '#cbd5e0';
          }
        }}
        onMouseOut={(e) => {
          if (currentHeight !== 1) {
            e.target.style.backgroundColor = '#e2e8f0';
          }
        }}
      >
        高度1
      </button>
      <button
        onClick={() => onHeightChange(2)}
        style={{
          backgroundColor: currentHeight === 2 ? '#10b981' : '#e2e8f0',
          color: currentHeight === 2 ? 'white' : '#4a5568',
          border: 'none',
          borderRadius: '6px',
          padding: '8px 16px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}
        onMouseOver={(e) => {
          if (currentHeight !== 2) {
            e.target.style.backgroundColor = '#cbd5e0';
          }
        }}
        onMouseOut={(e) => {
          if (currentHeight !== 2) {
            e.target.style.backgroundColor = '#e2e8f0';
          }
        }}
      >
        高度2
      </button>
    </div>
  );
};

// 时间轴组件（用于台风界面）
export const TyphoonTimeAxis = ({ currentTime, onTimeChange, showTime1Note = false }) => {
  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      right: '20px',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderRadius: '8px',
      padding: '12px 16px',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.15)',
      zIndex: 2001,
      display: 'flex',
      gap: '12px',
      alignItems: 'center',
      flexWrap: 'wrap'
    }}>
      <span style={{ fontSize: '14px', fontWeight: 600, color: '#2d3748', marginRight: '8px' }}>时间轴：</span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <button
          onClick={() => onTimeChange(1)}
          style={{
            backgroundColor: currentTime === 1 ? '#4299e1' : '#e2e8f0',
            color: currentTime === 1 ? 'white' : '#4a5568',
            border: 'none',
            borderRadius: '6px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: showTime1Note ? 'not-allowed' : 'pointer',
            transition: 'all 0.3s ease',
            opacity: showTime1Note ? 0.6 : 1,
            position: 'relative'
          }}
          onMouseOver={(e) => {
            if (currentTime !== 1 && !showTime1Note) {
              e.target.style.backgroundColor = '#cbd5e0';
            }
          }}
          onMouseOut={(e) => {
            if (currentTime !== 1 && !showTime1Note) {
              e.target.style.backgroundColor = '#e2e8f0';
            }
          }}
          disabled={showTime1Note}
          title={showTime1Note ? '时间1下无数据' : ''}
        >
          时间1
          {showTime1Note && (
            <span style={{
              fontSize: '10px',
              marginLeft: '4px',
              opacity: 0.8
            }}>(无数据)</span>
          )}
        </button>
        {showTime1Note && (
          <span style={{
            fontSize: '11px',
            color: '#dc2626',
            fontStyle: 'italic'
          }}>时间1下无数据</span>
        )}
      </div>
      <button
        onClick={() => onTimeChange(2)}
        style={{
          backgroundColor: currentTime === 2 ? '#4299e1' : '#e2e8f0',
          color: currentTime === 2 ? 'white' : '#4a5568',
          border: 'none',
          borderRadius: '6px',
          padding: '8px 16px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}
        onMouseOver={(e) => {
          if (currentTime !== 2) {
            e.target.style.backgroundColor = '#cbd5e0';
          }
        }}
        onMouseOut={(e) => {
          if (currentTime !== 2) {
            e.target.style.backgroundColor = '#e2e8f0';
          }
        }}
      >
        时间2
      </button>
    </div>
  );
};

// 生成模拟3D数据（高度场数据）
const generate3DData = (bounds, currentTime, currentHeight, resolution = 0.3) => {
  const data = [];
  const latSteps = Math.ceil((bounds.maxLat - bounds.minLat) / resolution);
  const lngSteps = Math.ceil((bounds.maxLng - bounds.minLng) / resolution);
  
  // 根据时间和高度调整数据模式
  const timeOffset = (currentTime - 1) * 0.5; // 时间偏移
  const heightMultiplier = currentHeight === 1 ? 1.0 : 1.3; // 高度2时数据值更大
  
  // 生成模拟的高度场数据（使用正弦波和余弦波创建起伏效果）
  for (let i = 0; i <= latSteps; i++) {
    const row = [];
    const lat = bounds.minLat + (i / latSteps) * (bounds.maxLat - bounds.minLat);
    for (let j = 0; j <= lngSteps; j++) {
      const lng = bounds.minLng + (j / lngSteps) * (bounds.maxLng - bounds.minLng);
      
      // 创建多个波叠加的3D效果（模拟地形或气压场）
      // 根据时间和高度调整波形
      const latNorm = (lat - bounds.minLat) / (bounds.maxLat - bounds.minLat);
      const lngNorm = (lng - bounds.minLng) / (bounds.maxLng - bounds.minLng);
      
      const value = 
        50 * heightMultiplier + 
        30 * heightMultiplier * Math.sin((latNorm * 2 + timeOffset) * Math.PI) * Math.cos((lngNorm * 2 + timeOffset) * Math.PI) +
        20 * heightMultiplier * Math.sin((latNorm * 4 + timeOffset * 0.5) * Math.PI) * Math.cos((lngNorm * 4 + timeOffset * 0.5) * Math.PI) +
        15 * heightMultiplier * Math.sin((latNorm * 6 + timeOffset * 0.3) * Math.PI) * Math.cos((lngNorm * 6 + timeOffset * 0.3) * Math.PI);
      
      row.push({
        lat,
        lng,
        value: Math.max(0, Math.min(100, value)) // 限制在0-100范围内
      });
    }
    data.push(row);
  }
  
  return data;
};

// 生成模拟盐度数据（基于demo.py的数据结构）- 3D体积数据
const generateSaltData3D = (bounds, currentTime, resolution = 0.5) => {
  // 生成3D网格数据：lat x lon x depth (8层)
  const latSteps = Math.ceil((bounds.maxLat - bounds.minLat) / resolution);
  const lngSteps = Math.ceil((bounds.maxLng - bounds.minLng) / resolution);
  const depthLayers = 8; // 8个深度层，对应0-1000米
  
  const data3D = [];
  const baseSalt = 35.0; // 基础盐度
  const timeOffset = (currentTime - 1) * 0.3; // 时间偏移
  
  // 生成3D体积数据
  for (let k = 0; k < depthLayers; k++) {
    const depth = (k / (depthLayers - 1)) * 1000; // 0-1000米
    const depthNorm = k / (depthLayers - 1); // 0-1
    const layer = [];
    
    for (let i = 0; i <= latSteps; i++) {
      const row = [];
      const lat = bounds.minLat + (i / latSteps) * (bounds.maxLat - bounds.minLat);
      
      for (let j = 0; j <= lngSteps; j++) {
        const lng = bounds.minLng + (j / lngSteps) * (bounds.maxLng - bounds.minLng);
        
        // 创建盐度分布模式（模拟海洋盐度变化）
        const latNorm = (lat - bounds.minLat) / (bounds.maxLat - bounds.minLat);
        const lngNorm = (lng - bounds.minLng) / (bounds.maxLng - bounds.minLng);
        
        // 盐度随纬度、经度、深度和时间变化
        const saltValue = 
          baseSalt + 
          2.0 * Math.sin((latNorm * 3 + timeOffset) * Math.PI) * Math.cos((lngNorm * 3 + timeOffset) * Math.PI) +
          1.5 * Math.sin((latNorm * 5 + timeOffset * 0.7) * Math.PI) * Math.cos((lngNorm * 5 + timeOffset * 0.7) * Math.PI) +
          (depthNorm - 0.5) * 1.0 + // 深度影响
          Math.sin(depthNorm * Math.PI * 2) * 0.5; // 深度波动
        
        row.push({
          lat,
          lng,
          depth,
          depthIndex: k,
          value: Math.max(30, Math.min(40, saltValue)) // 限制在30-40 g/kg范围内
        });
      }
      layer.push(row);
    }
    data3D.push(layer);
  }
  
  return data3D; // 返回3D数组：[depth][lat][lon]
};

// 将颜色字符串转换为RGBA格式
const colorToRGBA = (color, opacity) => {
  // 如果已经是rgba格式，提取RGB值
  if (color.startsWith('rgb')) {
    const match = color.match(/\d+/g);
    if (match && match.length >= 3) {
      return `rgba(${match[0]}, ${match[1]}, ${match[2]}, ${opacity})`;
    }
  }
  // 如果是hex格式，转换为rgba
  if (color.startsWith('#')) {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  }
  return color;
};

// 颜色映射函数（将数值映射到颜色）
const valueToColor = (value, min = 0, max = 100, isSalt = false) => {
  if (isSalt) {
    // 盐度数据使用viridis配色（类似demo.py）
    const normalized = (value - min) / (max - min);
    // viridis配色方案：从深紫到黄色
    if (normalized < 0.25) {
      const t = normalized / 0.25;
      return `rgb(${Math.round(68 + (72 - 68) * t)}, ${Math.round(1 + (40 - 1) * t)}, ${Math.round(84 + (120 - 84) * t)})`;
    } else if (normalized < 0.5) {
      const t = (normalized - 0.25) / 0.25;
      return `rgb(${Math.round(59 + (33 - 59) * t)}, ${Math.round(82 + (144 - 82) * t)}, ${Math.round(139 + (140 - 139) * t)})`;
    } else if (normalized < 0.75) {
      const t = (normalized - 0.5) / 0.25;
      return `rgb(${Math.round(33 + (92 - 33) * t)}, ${Math.round(144 + (200 - 144) * t)}, ${Math.round(140 + (141 - 140) * t)})`;
    } else {
      const t = (normalized - 0.75) / 0.25;
      return `rgb(${Math.round(92 + (253 - 92) * t)}, ${Math.round(200 + (231 - 200) * t)}, ${Math.round(141 + (37 - 141) * t)})`;
    }
  } else {
    // 高度场数据使用蓝色到红色的渐变
    const normalized = (value - min) / (max - min);
    if (normalized < 0.33) {
      const t = normalized / 0.33;
      return `rgb(${Math.round(48 + (65 - 48) * t)}, ${Math.round(18 + (69 - 18) * t)}, ${Math.round(59 + (171 - 59) * t)})`;
    } else if (normalized < 0.66) {
      const t = (normalized - 0.33) / 0.33;
      return `rgb(${Math.round(44 + (41 - 44) * t)}, ${Math.round(146 + (213 - 146) * t)}, ${Math.round(240 + (156 - 240) * t)})`;
    } else {
      const t = (normalized - 0.66) / 0.34;
      return `rgb(${Math.round(220 + (252 - 220) * t)}, ${Math.round(226 + (165 - 226) * t)}, ${Math.round(31 + (10 - 31) * t)})`;
    }
  }
};

// API配置
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// 从后端获取体积数据
const fetchVolumeData = async (timeIndex) => {
  try {
    console.log(`Fetching volume data from ${API_BASE_URL}/api/data/volume?time=${timeIndex}`);
    const response = await fetch(`${API_BASE_URL}/api/data/volume?time=${timeIndex}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Received response:', result);
    
    if (result.success) {
      return result;
    } else {
      throw new Error(result.error || 'Failed to fetch volume data');
    }
  } catch (error) {
    console.error('Error fetching volume data:', error);
    
    // 提供更详细的错误信息
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      throw new Error(`无法连接到后端服务器 (${API_BASE_URL})。请确保后端服务正在运行。`);
    } else if (error.message.includes('HTTP error')) {
      throw new Error(`后端服务器返回错误: ${error.message}`);
    } else {
      throw new Error(error.message || '加载数据失败');
    }
  }
};

// 3D立方体组件
const Cube3D = ({ typhoonId, currentTime, currentHeight, isVisible, onClose, center, useSimulation=true }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const rotationRef = useRef({ x: -0.5, y: 0.5 });
  const [volumeData, setVolumeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 获取网格范围（优先使用后端返回的 bounds）
  const getGridBounds = (id) => {
    if (volumeData && volumeData.bounds) {
      return {
        minLat: volumeData.bounds.minLat,
        maxLat: volumeData.bounds.maxLat,
        minLng: volumeData.bounds.minLng,
        maxLng: volumeData.bounds.maxLng
      };
    }
    // fallback ranges
    if (id === 1) return { minLat: 10, maxLat: 25, minLng: 105, maxLng: 125 };
    return { minLat: 0, maxLat: 40, minLng: 100, maxLng: 140 };
  };

  // 3D点旋转函数
  const rotatePoint = (point, rx, ry) => {
    // 绕X轴旋转
    let y = point.y;
    let z = point.z;
    point.y = y * Math.cos(rx) - z * Math.sin(rx);
    point.z = y * Math.sin(rx) + z * Math.cos(rx);
    
    // 绕Y轴旋转
    let x = point.x;
    z = point.z;
    point.x = x * Math.cos(ry) + z * Math.sin(ry);
    point.z = -x * Math.sin(ry) + z * Math.cos(ry);
  };

  // 投影3D点到2D
  const project = (point, distance) => {
    const fov = distance;
    const scale = fov / (fov + point.z);
    return {
      x: point.x * scale,
      y: point.y * scale,
      scale: scale
    };
  };

  // 绘制立方体
  const drawCube = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const bounds = getGridBounds(typhoonId);
    if (!bounds) return;

    // 立方体大小
    const size = 200;
    const centerX = width / 2;
    const centerY = height / 2;
    const distance = 500;

    // 定义立方体的8个顶点
    const vertices = [
      { x: -size, y: -size, z: -size }, // 0: 前左下
      { x: size, y: -size, z: -size },  // 1: 前右下
      { x: size, y: size, z: -size },   // 2: 前右上
      { x: -size, y: size, z: -size },  // 3: 前左上
      { x: -size, y: -size, z: size },  // 4: 后左下
      { x: size, y: -size, z: size },   // 5: 后右下
      { x: size, y: size, z: size },    // 6: 后右上
      { x: -size, y: size, z: size }     // 7: 后左上
    ];

    // 复制顶点并旋转
    const rotatedVertices = vertices.map(v => {
      const point = { ...v };
      rotatePoint(point, rotationRef.current.x, rotationRef.current.y);
      point.z += distance;
      return point;
    });

    // 投影到2D
    const projected = rotatedVertices.map(v => project(v, distance));

    // 定义立方体的面（6个面）
    const faces = [
      { indices: [0, 1, 2, 3], color: 'rgba(66, 153, 225, 0.3)' }, // 前面
      { indices: [4, 7, 6, 5], color: 'rgba(66, 153, 225, 0.2)' }, // 后面
      { indices: [0, 4, 5, 1], color: 'rgba(16, 185, 129, 0.3)' }, // 底面
      { indices: [2, 6, 7, 3], color: 'rgba(16, 185, 129, 0.2)' }, // 顶面
      { indices: [0, 3, 7, 4], color: 'rgba(239, 68, 68, 0.3)' }, // 左面
      { indices: [1, 5, 6, 2], color: 'rgba(239, 68, 68, 0.2)' }  // 右面
    ];

    // 计算每个面的深度（用于排序）
    const faceDepths = faces.map((face, i) => {
      const zSum = face.indices.reduce((sum, idx) => sum + rotatedVertices[idx].z, 0);
      return { index: i, depth: zSum / face.indices.length };
    });

    // 按深度排序（从后到前）
    faceDepths.sort((a, b) => b.depth - a.depth);

    // 绘制面
    faceDepths.forEach(({ index }) => {
      const face = faces[index];
      ctx.beginPath();
      const firstPoint = projected[face.indices[0]];
      ctx.moveTo(centerX + firstPoint.x, centerY + firstPoint.y);
      
      for (let i = 1; i < face.indices.length; i++) {
        const point = projected[face.indices[i]];
        ctx.lineTo(centerX + point.x, centerY + point.y);
      }
      ctx.closePath();

      ctx.fillStyle = face.color;
      ctx.fill();
      ctx.strokeStyle = 'rgba(66, 153, 225, 0.6)';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // 绘制边
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0], // 前面
      [4, 5], [5, 6], [6, 7], [7, 4], // 后面
      [0, 4], [1, 5], [2, 6], [3, 7]  // 连接前后
    ];

    ctx.strokeStyle = 'rgba(66, 153, 225, 0.8)';
    ctx.lineWidth = 2;
    edges.forEach(([start, end]) => {
      const p1 = projected[start];
      const p2 = projected[end];
      ctx.beginPath();
      ctx.moveTo(centerX + p1.x, centerY + p1.y);
      ctx.lineTo(centerX + p2.x, centerY + p2.y);
      ctx.stroke();
    });

    // 绘制轴标签
    ctx.fillStyle = '#2d3748';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    
    // X轴（经度）- 红色
    const xAxisEnd = projected[1]; // 前右下角
    ctx.fillStyle = '#ef4444';
    ctx.fillText('经度', centerX + xAxisEnd.x + 30, centerY + xAxisEnd.y);
    
    // Y轴（纬度）- 绿色
    const yAxisEnd = projected[3]; // 前左上角
    ctx.fillStyle = '#10b981';
    ctx.fillText('纬度', centerX + yAxisEnd.x, centerY + yAxisEnd.y - 30);
    
    // Z轴（高度）- 蓝色
    const zAxisEnd = projected[7]; // 后左上角
    ctx.fillStyle = '#4299e1';
    ctx.fillText('高度', centerX + zAxisEnd.x, centerY + zAxisEnd.y);

    // 绘制轴线和刻度
    const origin = projected[0]; // 原点（前左下角）
    
    // X轴（经度）- 红色
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX + origin.x, centerY + origin.y);
    ctx.lineTo(centerX + xAxisEnd.x, centerY + xAxisEnd.y);
    ctx.stroke();
    
    // X轴刻度（经度）
    const xAxisDir = {
      x: xAxisEnd.x - origin.x,
      y: xAxisEnd.y - origin.y
    };
    const xAxisLength = Math.sqrt(xAxisDir.x * xAxisDir.x + xAxisDir.y * xAxisDir.y);
    const xAxisUnit = { x: xAxisDir.x / xAxisLength, y: xAxisDir.y / xAxisLength };
    const xAxisPerp = { x: -xAxisUnit.y, y: xAxisUnit.x }; // 垂直方向
    
    // 经度刻度：从minLng到maxLng，每5度一个刻度
    for (let lng = bounds.minLng; lng <= bounds.maxLng; lng += 5) {
      const t = (lng - bounds.minLng) / (bounds.maxLng - bounds.minLng);
      const tickPos = {
        x: origin.x + xAxisDir.x * t,
        y: origin.y + xAxisDir.y * t
      };
      const tickRotated = { ...tickPos };
      rotatePoint(tickRotated, rotationRef.current.x, rotationRef.current.y);
      tickRotated.z += distance;
      const tickProj = project(tickRotated, distance);
      
      // 绘制刻度线
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX + tickProj.x, centerY + tickProj.y);
      const tickEnd = {
        x: tickProj.x + xAxisPerp.x * 8,
        y: tickProj.y + xAxisPerp.y * 8
      };
      ctx.lineTo(centerX + tickEnd.x, centerY + tickEnd.y);
      ctx.stroke();
      
      // 绘制数字标签
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${lng}°E`,
        centerX + tickEnd.x + xAxisPerp.x * 15,
        centerY + tickEnd.y + xAxisPerp.y * 15 + 4
      );
    }
    
    // Y轴（纬度）- 绿色
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX + origin.x, centerY + origin.y);
    ctx.lineTo(centerX + yAxisEnd.x, centerY + yAxisEnd.y);
    ctx.stroke();
    
    // Y轴刻度（纬度）
    const yAxisDir = {
      x: yAxisEnd.x - origin.x,
      y: yAxisEnd.y - origin.y
    };
    const yAxisLength = Math.sqrt(yAxisDir.x * yAxisDir.x + yAxisDir.y * yAxisDir.y);
    const yAxisUnit = { x: yAxisDir.x / yAxisLength, y: yAxisDir.y / yAxisLength };
    const yAxisPerp = { x: -yAxisUnit.y, y: yAxisUnit.x };
    
    // 纬度刻度：从minLat到maxLat，每5度一个刻度
    for (let lat = bounds.minLat; lat <= bounds.maxLat; lat += 5) {
      const t = (lat - bounds.minLat) / (bounds.maxLat - bounds.minLat);
      const tickPos = {
        x: origin.x + yAxisDir.x * t,
        y: origin.y + yAxisDir.y * t
      };
      const tickRotated = { ...tickPos };
      rotatePoint(tickRotated, rotationRef.current.x, rotationRef.current.y);
      tickRotated.z += distance;
      const tickProj = project(tickRotated, distance);
      
      // 绘制刻度线
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX + tickProj.x, centerY + tickProj.y);
      const tickEnd = {
        x: tickProj.x + yAxisPerp.x * 8,
        y: tickProj.y + yAxisPerp.y * 8
      };
      ctx.lineTo(centerX + tickEnd.x, centerY + tickEnd.y);
      ctx.stroke();
      
      // 绘制数字标签
      ctx.fillStyle = '#10b981';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${lat}°N`,
        centerX + tickEnd.x + yAxisPerp.x * 15,
        centerY + tickEnd.y + yAxisPerp.y * 15 + 4
      );
    }
    
    // Z轴（高度）- 蓝色
    ctx.strokeStyle = '#4299e1';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX + origin.x, centerY + origin.y);
    ctx.lineTo(centerX + zAxisEnd.x, centerY + zAxisEnd.y);
    ctx.stroke();
    
    // Z轴刻度（高度/深度）
    const zAxisDir = {
      x: zAxisEnd.x - origin.x,
      y: zAxisEnd.y - origin.y
    };
    const zAxisLength = Math.sqrt(zAxisDir.x * zAxisDir.x + zAxisDir.y * zAxisDir.y);
    const zAxisUnit = { x: zAxisDir.x / zAxisLength, y: zAxisDir.y / zAxisLength };
    const zAxisPerp = { x: -zAxisUnit.y, y: zAxisUnit.x };
    
    // 台风2显示深度（0-1000米，8层），台风1显示高度1和高度2
    const isTyphoon2Depth = typhoonId === 2;
    if (isTyphoon2Depth) {
      // 显示深度刻度：0-1000米，8层
      const depths = [0, 125, 250, 375, 500, 625, 750, 875, 1000];
      depths.forEach((depth, idx) => {
        const t = idx / (depths.length - 1); // 0到1
        const tickPos = {
          x: origin.x + zAxisDir.x * t,
          y: origin.y + zAxisDir.y * t
        };
        const tickRotated = { ...tickPos };
        rotatePoint(tickRotated, rotationRef.current.x, rotationRef.current.y);
        tickRotated.z += distance;
        const tickProj = project(tickRotated, distance);
        
        // 绘制刻度线
        ctx.strokeStyle = '#4299e1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX + tickProj.x, centerY + tickProj.y);
        const tickEnd = {
          x: tickProj.x + zAxisPerp.x * 8,
          y: tickProj.y + zAxisPerp.y * 8
        };
        ctx.lineTo(centerX + tickEnd.x, centerY + tickEnd.y);
        ctx.stroke();
        
        // 绘制数字标签（只显示部分深度值，避免过于拥挤）
        if (idx % 2 === 0 || idx === depths.length - 1) {
          ctx.fillStyle = '#4299e1';
          ctx.font = 'bold 12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(
            `${depth}m`,
            centerX + tickEnd.x + zAxisPerp.x * 15,
            centerY + tickEnd.y + zAxisPerp.y * 15 + 4
          );
        }
      });
    } else {
      // 高度刻度：高度1和高度2
      for (let h = 1; h <= 2; h++) {
        const t = (h - 1) / (2 - 1); // 0到1
        const tickPos = {
          x: origin.x + zAxisDir.x * t,
          y: origin.y + zAxisDir.y * t
        };
        const tickRotated = { ...tickPos };
        rotatePoint(tickRotated, rotationRef.current.x, rotationRef.current.y);
        tickRotated.z += distance;
        const tickProj = project(tickRotated, distance);
        
        // 绘制刻度线
        ctx.strokeStyle = '#4299e1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX + tickProj.x, centerY + tickProj.y);
        const tickEnd = {
          x: tickProj.x + zAxisPerp.x * 8,
          y: tickProj.y + zAxisPerp.y * 8
        };
        ctx.lineTo(centerX + tickEnd.x, centerY + tickEnd.y);
        ctx.stroke();
        
        // 绘制数字标签
        ctx.fillStyle = '#4299e1';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(
          `高度${h}`,
          centerX + tickEnd.x + zAxisPerp.x * 15,
          centerY + tickEnd.y + zAxisPerp.y * 15 + 4
        );
      }
    }

    // 在立方体内部绘制数据
    // 台风2使用盐度体积数据（demo.py），台风1使用高度场数据
    const isTyphoon2Data = typhoonId === 2;
    const latRange = bounds.maxLat - bounds.minLat;
    const lngRange = bounds.maxLng - bounds.minLng;
    const depthLayers = 8; // 8个深度层
    const heightRange = depthLayers; // 深度范围

    // 盐度数据的值范围
    const valueMin = isTyphoon2Data ? 30 : 0;
    const valueMax = isTyphoon2Data ? 40 : 100;

    if (isTyphoon2Data) {
      // 台风2：优先使用后端API数据，失败时使用模拟数据进行体积渲染
      let dataToRender = null;
      let useMockData = false;
      
      if (volumeData && volumeData.data) {
        // 使用后端真实数据
        dataToRender = volumeData.data;
        console.log('[Render] Using backend data for volume rendering');
      } else if (!loading && !error) {
        // 后端连接失败，使用模拟数据
        useMockData = true;
        const mockData3D = generateSaltData3D(bounds, currentTime, 0.5);
        dataToRender = mockData3D;
        console.log('[Render] Using mock data for volume rendering (backend unavailable)');
      }
      
      if (dataToRender) {
        const depthLayersCount = dataToRender.length;
        
        // 从后到前渲染每一层（体积渲染，类似demo.py的add_volume）
        for (let k = depthLayersCount - 1; k >= 0; k--) {
          const layer = dataToRender[k];
          const depthNorm = k / (depthLayersCount - 1);
          
          layer.forEach((row, i) => {
            if (i % 2 !== 0) return; // 采样，减少点数
            row.forEach((point, j) => {
              if (j % 2 !== 0) return; // 采样
              
              // 将经纬度和深度映射到立方体坐标
              const x = -size + ((point.lng - bounds.minLng) / lngRange) * size * 2;
              const y = size - ((point.lat - bounds.minLat) / latRange) * size * 2;
              const z = -size + (depthNorm) * size * 2;
              
              const dataPoint = { x, y, z };
              rotatePoint(dataPoint, rotationRef.current.x, rotationRef.current.y);
              dataPoint.z += distance;
              
              const proj = project(dataPoint, distance);
              if (proj.scale > 0 && proj.scale < 2) {
                // 使用sigmoid透明度函数（类似demo.py的opacity="sigmoid"）
                const normalized = (point.value - valueMin) / (valueMax - valueMin);
                const sigmoidOpacity = 1 / (1 + Math.exp(-10 * (normalized - 0.5)));
                const opacity = Math.min(0.15, sigmoidOpacity * 0.2); // 体积渲染透明度
                
                const color = valueToColor(point.value, valueMin, valueMax, true);
                const rgba = colorToRGBA(color, opacity);
                
                ctx.fillStyle = rgba;
                ctx.beginPath();
                // 使用稍大的点来模拟体积效果
                ctx.arc(centerX + proj.x, centerY + proj.y, 4 * proj.scale, 0, Math.PI * 2);
                ctx.fill();
              }
            });
          });
        }
        
        // 如果使用模拟数据，在角落显示提示
        if (useMockData) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
          ctx.font = '12px Arial';
          ctx.textAlign = 'left';
          ctx.fillText('使用模拟数据', 20, canvas.height - 20);
        }
      } else if (loading) {
        // 显示加载提示
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('正在加载数据...', centerX, centerY);
        ctx.font = '14px Arial';
        ctx.fillText('请稍候', centerX, centerY + 30);
      } else if (error) {
        // 显示错误提示（这种情况不应该出现，因为我们已经fallback到模拟数据）
        ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
        ctx.font = 'bold 18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('数据加载失败', centerX, centerY - 20);
        ctx.font = '14px Arial';
        ctx.fillStyle = 'rgba(255, 200, 200, 0.9)';
        const errorLines = error.split('。');
        errorLines.forEach((line, index) => {
          if (line.trim()) {
            ctx.fillText(line.trim(), centerX, centerY + index * 20);
          }
        });
      }
    } else {
      // 台风1：使用高度场数据（点云）
      const dataPoints = generate3DData(bounds, currentTime, currentHeight, 1.0);
      const heightRange2 = 2; // 高度范围（高度1和高度2）

      dataPoints.forEach((row, i) => {
        if (i % 3 !== 0) return; // 采样
        row.forEach((point, j) => {
          if (j % 3 !== 0) return; // 采样
          
          // 将经纬度映射到立方体坐标
          const x = -size + ((point.lng - bounds.minLng) / lngRange) * size * 2;
          const y = size - ((point.lat - bounds.minLat) / latRange) * size * 2;
          const z = -size + ((currentHeight - 1) / heightRange2) * size * 2;
          
          const dataPoint = { x, y, z };
          rotatePoint(dataPoint, rotationRef.current.x, rotationRef.current.y);
          dataPoint.z += distance;
          
          const proj = project(dataPoint, distance);
          if (proj.scale > 0) {
            const color = valueToColor(point.value, valueMin, valueMax, false);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(centerX + proj.x, centerY + proj.y, 3 * proj.scale, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      });
    }
  }, [typhoonId, currentTime, currentHeight, volumeData, loading, error]);

  // 从后端获取数据，失败时使用模拟数据
  useEffect(() => {
    // Both typhoon 1 and 2 can use backend volume data if available
    if ((typhoonId === 2 || typhoonId === 1) && isVisible) {
      setLoading(true);
      setError(null);
      console.log(`Fetching data for typhoon ${typhoonId}, time ${currentTime}`);
      if (useSimulation) {
        // Build simulated volume data using grid bounds
        try {
          const bounds = getGridBounds(typhoonId);
          const simVolume = generateSaltData3D(bounds, currentTime);
          // shape: depth major [depth][lat][lon]
          const nx = Math.ceil((bounds.maxLat - bounds.minLat) / 0.5) + 1;
          const ny = Math.ceil((bounds.maxLng - bounds.minLng) / 0.5) + 1;
          const nz = simVolume.length || 8;
          const payload = { success: true, data: simVolume, bounds, shape: { nx, ny, nz } };
          setVolumeData(payload);
          setLoading(false);
        } catch (err) {
          console.warn('Failed generating simulated volume data', err);
          setVolumeData(null);
          setLoading(false);
        }
      } else {
        fetchVolumeData(currentTime - 1) // API使用0-based索引
          .then((data) => {
            console.log('Data loaded successfully from backend:', data);
            setVolumeData(data);
            setLoading(false);
          })
          .catch((err) => {
            const errorMessage = err.message || '加载数据失败';
            console.warn('Failed to load volume data from backend, using mock data:', err);
            setVolumeData(null);
            setError(null);
            setLoading(false);
          });
      }
    } else if ((typhoonId === 2 || typhoonId === 1) && !isVisible) {
      // 当3D视图关闭时，清除数据
      setVolumeData(null);
      setError(null);
      setLoading(false);
    }
    }, [typhoonId, currentTime, isVisible]);

  // 动画循环
  useEffect(() => {
    if (!isVisible) return;

    const animate = () => {
      drawCube();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isVisible, drawCube]);

  // 鼠标事件处理
  const handleMouseDown = useCallback((e) => {
    isDraggingRef.current = true;
    const rect = canvasRef.current.getBoundingClientRect();
    lastMousePosRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!isDraggingRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    const deltaX = currentX - lastMousePosRef.current.x;
    const deltaY = currentY - lastMousePosRef.current.y;

    rotationRef.current.y += deltaX * 0.01;
    rotationRef.current.x += deltaY * 0.01;

    lastMousePosRef.current = { x: currentX, y: currentY };
    drawCube();
  }, [drawCube]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  if (!isVisible) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      zIndex: 3000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column'
    }}>
      <button
        onClick={onClose}
        style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          backgroundColor: '#ef4444',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '10px 20px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          zIndex: 3001
        }}
      >
        关闭3D视图
      </button>
      
      <div style={{
        color: 'white',
        marginBottom: '20px',
        fontSize: '18px',
        fontWeight: 600
      }}>
        3D数据立方体 - 拖拽鼠标旋转
      </div>
      
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          border: '2px solid rgba(66, 153, 225, 0.5)',
          borderRadius: '8px',
          backgroundColor: '#1a202c',
          cursor: isDraggingRef.current ? 'grabbing' : 'grab'
        }}
      />
      
      <div style={{
        color: 'white',
        marginTop: '20px',
        fontSize: '14px',
        textAlign: 'center'
      }}>
        <div style={{ marginBottom: '8px' }}>
          <span style={{ color: '#ef4444' }}>●</span> 经度轴
          <span style={{ marginLeft: '20px', color: '#10b981' }}>●</span> 纬度轴
          <span style={{ marginLeft: '20px', color: '#4299e1' }}>●</span> 高度轴
        </div>
        <div style={{ fontSize: '12px', opacity: 0.8 }}>
          时间{currentTime} - 高度{currentHeight}
        </div>
      </div>
    </div>
  );
};

// Visita 查看器组件（OpenVisus Web Viewer）
const VisitaViewer = ({ isVisible, onClose, datasetUrl }) => {
  if (!isVisible) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: '#0f172a',
      zIndex: 3000,
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        backgroundColor: '#1e293b',
        padding: '12px 20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
      }}>
        <div style={{ color: '#f8fafc', fontSize: '16px', fontWeight: 600 }}>
          OpenVisus Visita 视图
        </div>
        <button
          onClick={onClose}
          style={{
            backgroundColor: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#dc2626'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#ef4444'}
        >
          关闭
        </button>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        {datasetUrl ? (
          <iframe
            src={datasetUrl}
            style={{
              width: '100%',
              height: '100%',
              border: 'none',
            }}
            title="OpenVisus Visita Viewer"
            allowFullScreen
          />
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#cbd5e1',
            fontSize: '16px',
          }}>
            正在加载 Visita 视图...
          </div>
        )}
      </div>
    </div>
  );
};

// WebGL 3D立方体渲染器
const Interactive3DCube = ({ meshData, onClose }) => {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const programRef = useRef(null);
  const rotationRef = useRef({ x: -0.5, y: 0.5 });
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const zoomRef = useRef(1.0);
  const renderRef = useRef(null);

  useEffect(() => {
    if (!meshData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      console.error('WebGL not supported');
      return;
    }
    glRef.current = gl;

    // 顶点着色器
    const vertexShaderSource = `
      attribute vec3 a_position;
      attribute float a_salt;
      uniform mat4 u_matrix;
      uniform float u_saltMin;
      uniform float u_saltMax;
      varying float v_salt;
      
      void main() {
        gl_Position = u_matrix * vec4(a_position, 1.0);
        v_salt = (a_salt - u_saltMin) / (u_saltMax - u_saltMin);
      }
    `;

    // 片段着色器（使用兼容的语法）
    const fragmentShaderSource = `
      precision mediump float;
      varying float v_salt;
      
      vec3 colormap(float t) {
        t = clamp(t, 0.0, 1.0);
        float idx = t * 6.0;
        int i = int(floor(idx));
        float f = idx - float(i);
        
        vec3 c0, c1;
        
        if (i == 0) {
          c0 = vec3(48.0/255.0, 18.0/255.0, 59.0/255.0);
          c1 = vec3(65.0/255.0, 69.0/255.0, 171.0/255.0);
        } else if (i == 1) {
          c0 = vec3(65.0/255.0, 69.0/255.0, 171.0/255.0);
          c1 = vec3(44.0/255.0, 146.0/255.0, 240.0/255.0);
        } else if (i == 2) {
          c0 = vec3(44.0/255.0, 146.0/255.0, 240.0/255.0);
          c1 = vec3(41.0/255.0, 213.0/255.0, 156.0/255.0);
        } else if (i == 3) {
          c0 = vec3(41.0/255.0, 213.0/255.0, 156.0/255.0);
          c1 = vec3(220.0/255.0, 226.0/255.0, 31.0/255.0);
        } else if (i == 4) {
          c0 = vec3(220.0/255.0, 226.0/255.0, 31.0/255.0);
          c1 = vec3(252.0/255.0, 165.0/255.0, 10.0/255.0);
        } else if (i == 5) {
          c0 = vec3(252.0/255.0, 165.0/255.0, 10.0/255.0);
          c1 = vec3(252.0/255.0, 93.0/255.0, 7.0/255.0);
        } else {
          return vec3(252.0/255.0, 93.0/255.0, 7.0/255.0);
        }
        
        return mix(c0, c1, f);
      }
      
      void main() {
        vec3 color = colormap(v_salt);
        gl_FragColor = vec4(color, 0.6);
      }
    `;

    // 编译着色器
    function createShader(gl, type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    }

    function createProgram(gl, vertexShader, fragmentShader) {
      if (!vertexShader || !fragmentShader) {
        console.error('Cannot create program: shaders are null');
        return null;
      }
      const program = gl.createProgram();
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
      }
      return program;
    }

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    
    if (!vertexShader || !fragmentShader) {
      console.error('Failed to compile shaders');
      if (vertexShader) gl.deleteShader(vertexShader);
      if (fragmentShader) gl.deleteShader(fragmentShader);
      return;
    }
    
    const program = createProgram(gl, vertexShader, fragmentShader);
    if (!program) {
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      return;
    }
    programRef.current = program;

    // 准备数据
    const vertices = meshData.vertices;
    const faces = meshData.faces;
    const bounds = meshData.bounds;
    const saltRange = meshData.salt_range;

    // 计算中心点和范围
    const centerX = (bounds.x_min + bounds.x_max) / 2;
    const centerY = (bounds.y_min + bounds.y_max) / 2;
    const centerZ = (bounds.z_min + bounds.z_max) / 2;
    const rangeX = bounds.x_max - bounds.x_min;
    const rangeY = bounds.y_max - bounds.y_min;
    const rangeZ = bounds.z_max - bounds.z_min;
    const maxRange = Math.max(rangeX, rangeY, rangeZ);

    // 创建顶点缓冲区（多层表面，类似PyVista）
    const positions = [];
    const salts = [];
    const layerOffsets = []; // 每层的起始索引和数量
    
    // 如果有layer_faces，按层渲染（更接近PyVista效果）
    if (meshData.layer_faces && meshData.layer_faces.length > 0) {
      for (let layerIdx = 0; layerIdx < meshData.layer_faces.length; layerIdx++) {
        const startIdx = positions.length / 3;
        const layerFaceIndices = meshData.layer_faces[layerIdx];
        
        for (const faceIdx of layerFaceIndices) {
          const face = faces[faceIdx];
          if (face && face.length >= 4) {
            // 将四边形分解为两个三角形
            // 三角形1: 0, 1, 2
            for (let i = 0; i < 3; i++) {
              const idx = face[i];
              const v = vertices[idx];
              positions.push(v[0] - centerX, v[1] - centerY, v[2] - centerZ);
              salts.push(v[3]);
            }
            // 三角形2: 0, 2, 3
            positions.push(vertices[face[0]][0] - centerX, vertices[face[0]][1] - centerY, vertices[face[0]][2] - centerZ);
            salts.push(vertices[face[0]][3]);
            positions.push(vertices[face[2]][0] - centerX, vertices[face[2]][1] - centerY, vertices[face[2]][2] - centerZ);
            salts.push(vertices[face[2]][3]);
            positions.push(vertices[face[3]][0] - centerX, vertices[face[3]][1] - centerY, vertices[face[3]][2] - centerZ);
            salts.push(vertices[face[3]][3]);
          }
        }
        const endIdx = positions.length / 3;
        layerOffsets.push({ start: startIdx, count: endIdx - startIdx, layer: layerIdx });
      }
    } else {
      // 回退到原来的方式（渲染所有面）
      for (const face of faces) {
        if (face.length >= 4) {
          for (let i = 0; i < 3; i++) {
            const idx = face[i];
            const v = vertices[idx];
            positions.push(v[0] - centerX, v[1] - centerY, v[2] - centerZ);
            salts.push(v[3]);
          }
          positions.push(vertices[face[0]][0] - centerX, vertices[face[0]][1] - centerY, vertices[face[0]][2] - centerZ);
          salts.push(vertices[face[0]][3]);
          positions.push(vertices[face[2]][0] - centerX, vertices[face[2]][1] - centerY, vertices[face[2]][2] - centerZ);
          salts.push(vertices[face[2]][3]);
          positions.push(vertices[face[3]][0] - centerX, vertices[face[3]][1] - centerY, vertices[face[3]][2] - centerZ);
          salts.push(vertices[face[3]][3]);
        }
      }
    }

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const saltBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, saltBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(salts), gl.STATIC_DRAW);
    
    // 存储layerOffsets供render函数使用
    const layerOffsetsRef = { current: layerOffsets };

    // 渲染函数
    const render = () => {
      const gl = glRef.current;
      if (!gl || !programRef.current) return;

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.1, 0.1, 0.15, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      gl.useProgram(programRef.current);

      // 设置矩阵（简化的正交投影 + 旋转）
      const aspect = canvas.width / canvas.height;
      const scale = zoomRef.current * 2.0 / maxRange;

      // 旋转角度
      const rx = rotationRef.current.x;
      const ry = rotationRef.current.y;
      const cosRx = Math.cos(rx);
      const sinRx = Math.sin(rx);
      const cosRy = Math.cos(ry);
      const sinRy = Math.sin(ry);

      // 组合矩阵：正交投影 * 旋转Y * 旋转X * 缩放
      // 先绕Y轴旋转，再绕X轴旋转
      const matrix = [
        scale * cosRy / aspect, scale * sinRx * sinRy / aspect, -scale * cosRx * sinRy, 0,
        -scale * sinRy / aspect, scale * sinRx * cosRy, scale * cosRx * cosRy, 0,
        0, scale * cosRx, scale * sinRx, 0,
        0, 0, 0, 1
      ];

      const matrixLocation = gl.getUniformLocation(programRef.current, 'u_matrix');
      gl.uniformMatrix4fv(matrixLocation, false, matrix);

      const saltMinLocation = gl.getUniformLocation(programRef.current, 'u_saltMin');
      const saltMaxLocation = gl.getUniformLocation(programRef.current, 'u_saltMax');
      gl.uniform1f(saltMinLocation, saltRange.min);
      gl.uniform1f(saltMaxLocation, saltRange.max);

      // 设置属性
      const positionLocation = gl.getAttribLocation(programRef.current, 'a_position');
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

      const saltLocation = gl.getAttribLocation(programRef.current, 'a_salt');
      gl.bindBuffer(gl.ARRAY_BUFFER, saltBuffer);
      gl.enableVertexAttribArray(saltLocation);
      gl.vertexAttribPointer(saltLocation, 1, gl.FLOAT, false, 0, 0);

      // 绘制（如果有分层数据，按层绘制，类似PyVista的多层叠加效果）
      if (layerOffsetsRef.current && layerOffsetsRef.current.length > 0) {
        // 从后到前绘制每一层（深度排序，类似PyVista的透明度叠加）
        for (let i = layerOffsetsRef.current.length - 1; i >= 0; i--) {
          const offset = layerOffsetsRef.current[i];
          gl.drawArrays(gl.TRIANGLES, offset.start, offset.count);
        }
      } else {
        // 绘制所有三角形
        gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
      }
    };
    
    // 存储render函数到ref
    renderRef.current = render;

    // 鼠标事件（仅旋转，不拖拽）
    const handleMouseDown = (e) => {
      isDraggingRef.current = true;
      lastMousePosRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
    };

    const handleMouseMove = (e) => {
      if (!isDraggingRef.current) {
        // 鼠标悬停时显示可旋转的提示
        canvas.style.cursor = 'grab';
        return;
      }
      const dx = e.clientX - lastMousePosRef.current.x;
      const dy = e.clientY - lastMousePosRef.current.y;
      rotationRef.current.y += dx * 0.01;
      rotationRef.current.x += dy * 0.01;
      lastMousePosRef.current = { x: e.clientX, y: e.clientY };
      render();
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      canvas.style.cursor = 'grab';
    };
    
    const handleMouseLeave = () => {
      isDraggingRef.current = false;
      canvas.style.cursor = 'default';
    };

    const handleWheel = (e) => {
      e.preventDefault();
      zoomRef.current *= (1 + e.deltaY * 0.001);
      zoomRef.current = Math.max(0.1, Math.min(5.0, zoomRef.current));
      render();
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    canvas.addEventListener('wheel', handleWheel);
    
    // 初始设置鼠标样式
    canvas.style.cursor = 'grab';

    // 初始渲染
    render();

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [meshData]);


  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        borderRadius: 8,
        display: 'block',
        userSelect: 'none' // 防止拖拽时选中文本
      }}
    />
  );
};

const Typhoon3Viewer = ({ isVisible, onClose, typhoonId = 3 }) => {
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generatedHint, setGeneratedHint] = useState(null);

  const fetchImage = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      // 支持所有台风ID，使用通用API
      const url = `${API_BASE_URL}/api/typhoon/${typhoonId}/image?ts=${Date.now()}`;
      console.log('[Typhoon3Viewer] Fetching image from:', url);
      const res = await fetch(url);
      
      // 检查响应内容类型
      const contentType = res.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await res.text();
        console.error('[Typhoon3Viewer] Non-JSON response:', text.substring(0, 200));
        throw new Error(`后端返回了非JSON响应 (${res.status})。请检查后端服务器是否正常运行。`);
      }
      
      const payload = await res.json();
      if (!res.ok || !payload.success) {
        throw new Error(payload.error || `HTTP ${res.status}`);
      }
      setImageData(payload.image);
      const now = new Date();
      setGeneratedHint(now.toLocaleString());
    } catch (err) {
      console.error('[Typhoon3Viewer] Error fetching image:', err);
      setError(err.message || '加载失败');
      setImageData(null);
    } finally {
      setLoading(false);
    }
  }, [typhoonId]);

  useEffect(() => {
    if (isVisible) {
      fetchImage();
    } else {
      setImageData(null);
      setError(null);
      setGeneratedHint(null);
    }
  }, [isVisible, fetchImage]);

  if (!isVisible) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      zIndex: 3000,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 20,
      boxSizing: 'border-box'
    }}>
      <button
        onClick={onClose}
        style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          backgroundColor: '#ef4444',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '10px 20px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          zIndex: 3001
        }}
      >
        关闭3D视图
      </button>
      <div style={{ color: '#f8fafc', marginBottom: 12, fontSize: 20, fontWeight: 600 }}>
        台风{typhoonId} · text.py PyVista 结果
      </div>
      <div style={{ color: '#cbd5f5', marginBottom: 32, fontSize: 14 }}>
        调用 `src/text.py` 离屏渲染（参数取自 `backend/config.json`），展示盐度+流线场
      </div>
      <div style={{
        width: '80%',
        maxWidth: 1000,
        minHeight: 520,
        backgroundColor: '#0f172a',
        borderRadius: 12,
        border: '2px solid rgba(148, 163, 184, 0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
        boxShadow: '0 10px 30px rgba(15, 23, 42, 0.6)'
      }}>
        {loading && (
          <div style={{ color: '#f1f5f9', fontSize: 16 }}>正在生成 PyVista 场景...</div>
        )}
        {!loading && error && (
          <div style={{ color: '#fecdd3', textAlign: 'center' }}>
            <p>渲染失败：{error}</p>
            <button
              onClick={fetchImage}
              style={{
                marginTop: 12,
                backgroundColor: '#f97316',
                color: 'white',
                border: 'none',
                borderRadius: 8,
                padding: '8px 16px',
                cursor: 'pointer'
              }}
            >
              重试
            </button>
          </div>
        )}
        {!loading && !error && imageData && (
          <img
            src={imageData}
            alt="Typhoon 3 visualization"
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              borderRadius: 8,
              border: '1px solid rgba(148, 163, 184, 0.4)',
              boxShadow: '0 6px 20px rgba(15, 23, 42, 0.8)'
            }}
          />
        )}
      </div>
      <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
        <button
          onClick={fetchImage}
          style={{
            backgroundColor: '#6366f1',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            padding: '10px 18px',
            fontSize: 14,
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(99,102,241,0.4)'
          }}
        >
          重新渲染 text.py
        </button>
        {generatedHint && (
          <div style={{ color: '#cbd5f5', fontSize: 12, alignSelf: 'center' }}>
            最近生成时间：{generatedHint}
          </div>
        )}
      </div>
    </div>
  );
};

// 盐度3D可视化的区域参数配置面板（原主界面 ConfigPanel，迁移到台风详情页面）
const Salt3DConfigPanel = ({ visible, onClose, onApply }) => {
  const [config, setConfig] = useState({
    lat_start: 10,
    lat_end: 40,
    lon_start: 100,
    lon_end: 130,
    nz: 20,
    data_quality: -6,
    scale_xy: 25,
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState({ error: null, success: null });

  const fieldStyle = {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    fontSize: '12px',
    color: '#4a5568',
  };

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE_URL}/api/config`);
      const payload = await res.json();
      if (!res.ok || !payload.success) {
        throw new Error(payload.error || `HTTP ${res.status}`);
      }
      setConfig(payload.config || config);
      setStatus({ error: null, success: null });
    } catch (err) {
      console.error('[Salt3DConfigPanel] Failed to load backend config', err);
      setStatus({ error: `无法读取配置：${err.message}`, success: null });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (visible) {
      fetchConfig();
    }
  }, [visible, fetchConfig]);

  const handleChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setStatus(prev => ({ ...prev, success: null }));
  };

  const handleSaveAndApply = async () => {
    try {
      setSaving(true);
      setStatus({ error: null, success: null });
      const res = await fetch(`${API_BASE_URL}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const payload = await res.json();
      if (!res.ok || !payload.success) {
        throw new Error(payload.error || `HTTP ${res.status}`);
      }
      setConfig(payload.config || config);
      setStatus({ error: null, success: '已保存区域参数，并开始生成盐度3D可视化。' });
      if (onApply) {
        onApply(payload.config || config);
      }
      onClose();
    } catch (err) {
      console.error('[Salt3DConfigPanel] Failed to save backend config', err);
      setStatus({ error: `保存失败：${err.message}`, success: null });
    } finally {
      setSaving(false);
    }
  };

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        zIndex: 3500,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        style={{
          background: 'rgba(255,255,255,0.97)',
          padding: 20,
          borderRadius: 12,
          boxShadow: '0 10px 30px rgba(15,23,42,0.4)',
          width: 380,
          maxWidth: '95vw',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <h3 style={{ margin: 0, color: '#1f2937' }}>盐度3D可视化区域参数</h3>
          <button
            onClick={onClose}
            style={{ border: 'none', background: 'transparent', fontSize: 20, cursor: 'pointer', color: '#4b5563' }}
          >
            ×
          </button>
        </div>

        {loading && (
          <p style={{ margin: '4px 0 12px', color: '#64748b', fontSize: 13 }}>正在读取当前配置...</p>
        )}
        {status.error && (
          <p style={{ margin: '4px 0 12px', color: '#dc2626', fontSize: 13 }}>{status.error}</p>
        )}
        {status.success && (
          <p style={{ margin: '4px 0 12px', color: '#16a34a', fontSize: 13 }}>{status.success}</p>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <label style={fieldStyle}>
            lat_start
            <input
              type="number"
              value={config.lat_start ?? ''}
              onChange={(e) => handleChange('lat_start', parseFloat(e.target.value))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
              step="0.1"
            />
          </label>
          <label style={fieldStyle}>
            lat_end
            <input
              type="number"
              value={config.lat_end ?? ''}
              onChange={(e) => handleChange('lat_end', parseFloat(e.target.value))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
              step="0.1"
            />
          </label>
          <label style={fieldStyle}>
            lon_start
            <input
              type="number"
              value={config.lon_start ?? ''}
              onChange={(e) => handleChange('lon_start', parseFloat(e.target.value))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
              step="0.1"
            />
          </label>
          <label style={fieldStyle}>
            lon_end
            <input
              type="number"
              value={config.lon_end ?? ''}
              onChange={(e) => handleChange('lon_end', parseFloat(e.target.value))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
              step="0.1"
            />
          </label>
          <label style={fieldStyle}>
            nz
            <input
              type="number"
              value={config.nz ?? ''}
              onChange={(e) => handleChange('nz', parseInt(e.target.value, 10))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
            />
          </label>
          <label style={fieldStyle}>
            scale_xy
            <input
              type="number"
              value={config.scale_xy ?? ''}
              onChange={(e) => handleChange('scale_xy', parseFloat(e.target.value))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
              step="1"
            />
          </label>
          <label style={fieldStyle}>
            data_quality
            <input
              type="number"
              value={config.data_quality ?? ''}
              onChange={(e) => handleChange('data_quality', parseInt(e.target.value, 10))}
              style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5f5' }}
            />
          </label>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 14px',
              borderRadius: 8,
              border: '1px solid #e2e8f0',
              background: 'white',
              color: '#4b5563',
              cursor: 'pointer',
            }}
          >
            取消
          </button>
          <button
            onClick={handleSaveAndApply}
            disabled={saving}
            style={{
              padding: '8px 16px',
              borderRadius: 8,
              border: 'none',
              background: saving ? '#94a3b8' : '#4299e1',
              color: 'white',
              cursor: saving ? 'not-allowed' : 'pointer',
              minWidth: 120,
            }}
          >
            {saving ? '保存中...' : '保存并生成'}
          </button>
        </div>

        <p style={{ marginTop: 10, fontSize: 12, color: '#64748b' }}>
          提示：保存后后端 `config.json` 将更新，`text.py` 与相关 API 会使用最新区域参数生成盐度3D可视化。
        </p>
      </div>
    </div>
  );
};

// 颜色图例组件
const ColorLegend = () => {
  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderRadius: '8px',
      padding: '12px 16px',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.15)',
      zIndex: 2001,
      minWidth: '200px'
    }}>
      <div style={{ fontSize: '12px', fontWeight: 600, color: '#2d3748', marginBottom: '8px' }}>
        3D数据值（高度场）
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: '150px',
          height: '20px',
          background: 'linear-gradient(to right, rgb(48,18,59), rgb(65,69,171), rgb(44,213,156), rgb(220,226,31), rgb(252,165,10))',
          borderRadius: '4px',
          border: '1px solid #cbd5e0'
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#4a5568', marginTop: '4px' }}>
        <span>低</span>
        <span>高</span>
      </div>
    </div>
  );
};

// 经纬度网格和3D数据展示组件
const LatLngGrid = ({ typhoonId, currentTime, currentHeight }) => {
  const canvasRef = useRef(null);
  const dataCanvasRef = useRef(null);

  // 根据台风ID确定网格范围
  const getGridBounds = (id) => {
    if (id === 1) {
      // 南海区域
      return {
        minLat: 10,
        maxLat: 25,
        minLng: 105,
        maxLng: 125,
        latStep: 2.5,  // 每2.5度一条纬线
        lngStep: 2.5   // 每2.5度一条经线
      };
    } else if (id === 2) {
      // 西太平洋区域
      return {
        minLat: 15,
        maxLat: 35,
        minLng: 120,
        maxLng: 160,
        latStep: 2.5,
        lngStep: 2.5
      };
    }
    return null;
  };

  // 绘制3D数据
  useEffect(() => {
    const dataCanvas = dataCanvasRef.current;
    if (!dataCanvas || typhoonId !== 1) return; // 只在台风1显示3D数据

    const bounds = getGridBounds(typhoonId);
    if (!bounds) return;

    const ctx = dataCanvas.getContext('2d');
    const width = dataCanvas.width;
    const height = dataCanvas.height;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    // 生成3D数据（根据时间和高度）
    const data = generate3DData(bounds, currentTime, currentHeight, 0.3);
    
    // 计算经纬度到像素的转换
    const latRange = bounds.maxLat - bounds.minLat;
    const lngRange = bounds.maxLng - bounds.minLng;

    // 绘制3D数据（使用颜色映射）
    const cellWidth = width / (data[0].length - 1);
    const cellHeight = height / (data.length - 1);

    for (let i = 0; i < data.length - 1; i++) {
      for (let j = 0; j < data[i].length - 1; j++) {
        // 获取四个角的值
        const v1 = data[i][j].value;
        const v2 = data[i][j + 1].value;
        const v3 = data[i + 1][j].value;
        const v4 = data[i + 1][j + 1].value;
        
        // 计算平均值
        const avgValue = (v1 + v2 + v3 + v4) / 4;
        
        // 转换为颜色
        const color = valueToColor(avgValue);
        
        // 绘制矩形
        const x = (data[i][j].lng - bounds.minLng) / lngRange * width;
        const y = height - (data[i][j].lat - bounds.minLat) / latRange * height;
        
        ctx.fillStyle = color;
        ctx.fillRect(x, y - cellHeight, cellWidth, cellHeight);
        // 绘制点数据（dataPoints）
    
        // （已移除在LatLngGrid中错误插入的 center 绘制，center 绘制应在 Cube3D 内完成）
          }
        }

    // 绘制等值线（可选）
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    
    const contourLevels = [20, 40, 60, 80];
    contourLevels.forEach(level => {
      ctx.beginPath();
      for (let i = 0; i < data.length - 1; i++) {
        for (let j = 0; j < data[i].length - 1; j++) {
          const v1 = data[i][j].value;
          const v2 = data[i][j + 1].value;
          const v3 = data[i + 1][j].value;
          const v4 = data[i + 1][j + 1].value;
          
          const x = (data[i][j].lng - bounds.minLng) / lngRange * width;
          const y = height - (data[i][j].lat - bounds.minLat) / latRange * height;
          
          // 简化的等值线绘制
          if ((v1 <= level && v2 > level) || (v1 > level && v2 <= level)) {
            if (!ctx.isPointInPath) {
              ctx.moveTo(x + cellWidth, y);
              ctx.lineTo(x + cellWidth, y - cellHeight);
            }
          }
        }
      }
      ctx.stroke();
    });
  }, [typhoonId, currentTime, currentHeight]);

  // 绘制经纬度网格
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const bounds = getGridBounds(typhoonId);
    if (!bounds) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    // 设置样式
    ctx.strokeStyle = 'rgba(66, 153, 225, 0.5)';
    ctx.lineWidth = 1;
    ctx.font = '11px Arial';
    ctx.fillStyle = '#2d3748';

    // 计算经纬度到像素的转换
    const latRange = bounds.maxLat - bounds.minLat;
    const lngRange = bounds.maxLng - bounds.minLng;

    // 绘制纬线（水平线）
    for (let lat = bounds.minLat; lat <= bounds.maxLat; lat += bounds.latStep) {
      const y = height - ((lat - bounds.minLat) / latRange) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();

      // 绘制纬度标签（左侧）
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#2d3748';
      ctx.lineWidth = 2;
      ctx.strokeText(`${lat.toFixed(1)}°N`, 5, y - 3);
      ctx.fillText(`${lat.toFixed(1)}°N`, 5, y - 3);
      // 绘制纬度标签（右侧）
      ctx.strokeText(`${lat.toFixed(1)}°N`, width - 50, y - 3);
      ctx.fillText(`${lat.toFixed(1)}°N`, width - 50, y - 3);
      ctx.fillStyle = '#2d3748';
      ctx.strokeStyle = 'rgba(66, 153, 225, 0.5)';
      ctx.lineWidth = 1;
    }

    // 绘制经线（垂直线）
    for (let lng = bounds.minLng; lng <= bounds.maxLng; lng += bounds.lngStep) {
      const x = ((lng - bounds.minLng) / lngRange) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      // 绘制经度标签（顶部）
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#2d3748';
      ctx.lineWidth = 2;
      ctx.strokeText(`${lng.toFixed(1)}°E`, x + 3, 15);
      ctx.fillText(`${lng.toFixed(1)}°E`, x + 3, 15);
      // 绘制经度标签（底部）
      ctx.strokeText(`${lng.toFixed(1)}°E`, x + 3, height - 5);
      ctx.fillText(`${lng.toFixed(1)}°E`, x + 3, height - 5);
      ctx.fillStyle = '#2d3748';
      ctx.strokeStyle = 'rgba(66, 153, 225, 0.5)';
      ctx.lineWidth = 1;
    }

    // 绘制边框
    ctx.strokeStyle = 'rgba(66, 153, 225, 0.8)';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, width, height);
  }, [typhoonId]);

  const bounds = getGridBounds(typhoonId);
  if (!bounds) return null;

  return (
    <div style={{
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      width: '70%',
      height: '60%',
      maxWidth: '900px',
      maxHeight: '600px',
      zIndex: 1999,
      pointerEvents: 'none'
    }}>
      {/* 3D数据层（仅在台风1显示） */}
      {typhoonId === 1 && (
        <>
          <canvas
            ref={dataCanvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              borderRadius: '8px',
              zIndex: 1998
            }}
            width={900}
            height={600}
          />
          <ColorLegend />
        </>
      )}
      
      {/* 经纬度网格层 */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          border: '2px solid rgba(66, 153, 225, 0.5)',
          borderRadius: '8px',
          zIndex: 1999
        }}
        width={900}
        height={600}
      />
    </div>
  );
};

// 台风界面组件
// 可视化历史记录浮标组件
export const VisualizationHistoryPanel = ({ isVisible, onClose, history, onClearHistory }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState(null); // 放大的图片

  // 添加 ESC 键监听
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && enlargedImage) {
        setEnlargedImage(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [enlargedImage]);

  // 条件返回必须在所有 hooks 之后
  if (!isVisible) return null;

  return (
    <>
      {/* 浮标按钮 */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          position: 'fixed',
          bottom: '30px',
          right: '30px',
          width: '60px',
          height: '60px',
          backgroundColor: '#8b5cf6',
          borderRadius: '50%',
          boxShadow: '0 4px 20px rgba(139, 92, 246, 0.4)',
          zIndex: 4000,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.3s ease',
          border: '3px solid white'
        }}
        onMouseOver={(e) => {
          e.currentTarget.style.backgroundColor = '#7c3aed';
          e.currentTarget.style.transform = 'scale(1.1)';
          e.currentTarget.style.boxShadow = '0 6px 25px rgba(139, 92, 246, 0.6)';
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.backgroundColor = '#8b5cf6';
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.boxShadow = '0 4px 20px rgba(139, 92, 246, 0.4)';
        }}
      >
        <span style={{ fontSize: '24px' }}>📊</span>
        {history.length > 0 && (
          <span style={{
            position: 'absolute',
            top: '-5px',
            right: '-5px',
            backgroundColor: '#ef4444',
            color: 'white',
            borderRadius: '50%',
            width: '24px',
            height: '24px',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 600,
            border: '2px solid white'
          }}>
            {history.length > 99 ? '99+' : history.length}
          </span>
        )}
      </div>

      {/* 展开的历史记录面板 */}
      {isExpanded && (
        <div style={{
          position: 'fixed',
          bottom: '110px',
          right: '30px',
          width: '420px',
          maxHeight: 'calc(100vh - 150px)',
          backgroundColor: '#ffffff',
          borderRadius: 16,
          padding: 24,
          boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
          zIndex: 4001,
          overflow: 'auto',
          animation: 'slideUp 0.3s ease-out'
        }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: '18px' }}>可视化历史</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          {history.length > 0 && (
            <button onClick={onClearHistory} style={{
              backgroundColor: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              padding: '6px 12px',
              cursor: 'pointer',
              fontSize: 12
            }}>清空</button>
          )}
          <button onClick={() => setIsExpanded(false)} style={{
            backgroundColor: '#6b7280',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            padding: '6px 12px',
            cursor: 'pointer',
            fontSize: 12
          }}>关闭</button>
        </div>
      </div>

      {history.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '40px',
          color: '#9ca3af',
          fontSize: '14px'
        }}>
          暂无历史记录
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {history.map((item, index) => (
            <div
              key={index}
              style={{
                border: '1px solid #e5e7eb',
                borderRadius: 8,
                padding: 12,
                backgroundColor: '#f9fafb',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor = '#f3f4f6';
                e.currentTarget.style.borderColor = '#10b981';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = '#f9fafb';
                e.currentTarget.style.borderColor = '#e5e7eb';
              }}
            >
              {(() => {
                // 兼容后端仅返回base64时的显示
                if (item.image && !item.image.startsWith('data:')) {
                  item.image = `data:image/png;base64,${item.image}`;
                }
              })()}
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: '14px', fontWeight: 600, color: '#1f2937', marginBottom: 4 }}>
                  {item.method}
                </div>
                <div style={{ fontSize: '12px', color: '#6b7280' }}>
                  时间步: {item.timeStep} | {new Date(item.timestamp).toLocaleString('zh-CN')}
                </div>
              </div>
              {item.image && (
                <img
                  src={item.image}
                  alt={item.method}
                  style={{
                    width: '100%',
                    borderRadius: 6,
                    border: '1px solid #e5e7eb',
                    marginTop: 8,
                    cursor: 'pointer'
                  }}
                  onDoubleClick={() => setEnlargedImage(item.image)}
                  title="双击放大图片"
                />
              )}
            </div>
          ))}
        </div>
      )}
      </div>
      )}

      {/* 遮罩层，点击关闭面板 */}
      {isExpanded && (
        <div
          onClick={() => setIsExpanded(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            zIndex: 3999,
            cursor: 'pointer'
          }}
        />
      )}

      {/* 放大图片弹窗 */}
      {enlargedImage && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            zIndex: 5000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer'
          }}
          onClick={() => setEnlargedImage(null)}
        >
          <div
            style={{
              position: 'relative',
              maxWidth: '95vw',
              maxHeight: '95vh',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={enlargedImage}
              alt="放大图片"
              style={{
                maxWidth: '95vw',
                maxHeight: '95vh',
                borderRadius: 8,
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)'
              }}
            />
            <button
              onClick={() => setEnlargedImage(null)}
              style={{
                position: 'absolute',
                top: '-40px',
                right: 0,
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                color: '#1f2937',
                border: 'none',
                borderRadius: 6,
                padding: '8px 16px',
                cursor: 'pointer',
                fontSize: 14,
                fontWeight: 600,
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = '#ffffff';
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
              }}
            >
              关闭 (ESC)
            </button>
          </div>
        </div>
      )}
    </>
  );
};

// 3D可视化组件（海洋盐度3D可视化，对应主界面“区域参数”配置）
const Velocity3DPanel = ({ isVisible, onClose, onSaveHistory, currentTimeStep }) => {
  const [strategyIdx, setStrategyIdx] = useState(1);
  const [vectorMode, setVectorMode] = useState(1);
  const [latStart, setLatStart] = useState(10);
  const [latEnd, setLatEnd] = useState(40);
  const [lonStart, setLonStart] = useState(100);
  const [lonEnd, setLonEnd] = useState(130);
  const [nz, setNz] = useState(10);
  const [dataQuality, setDataQuality] = useState(-6);
  const [scaleXy, setScaleXy] = useState(25);
  const [skip, setSkip] = useState('');
  const [arrowScale, setArrowScale] = useState(60.0);
  const [kNeighbors, setKNeighbors] = useState(4);
  const [maxBendFactor, setMaxBendFactor] = useState(0.3);
  const [streamlineLength, setStreamlineLength] = useState(50.0);
  const [stepSize, setStepSize] = useState(0.5);
  const [nSeeds, setNSeeds] = useState(400);
  const [targetClusters, setTargetClusters] = useState(20);
  const [loading, setLoading] = useState(false);
  const [resultImage, setResultImage] = useState(null);
  const [error, setError] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [showConfig, setShowConfig] = useState(true); // 控制参数配置面板的显示
  const [renderMode, setRenderMode] = useState('image'); // 'image' 或 'window'

  // 加载策略列表
  useEffect(() => {
    if (isVisible) {
      fetch(`${API_BASE_URL}/api/velocity3d/strategies`)
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setStrategies(data.strategies);
          }
        })
        .catch(err => console.error('加载策略列表失败:', err));
    }
  }, [isVisible]);

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      setResultImage(null);

      const params = {
        strategy_idx: strategyIdx,
        vector_mode: vectorMode,
        lat_start: latStart,
        lat_end: latEnd,
        lon_start: lonStart,
        lon_end: lonEnd,
        nz: nz,
        data_quality: dataQuality,
        scale_xy: scaleXy,
        arrow_scale: arrowScale,
        k_neighbors: kNeighbors,
        max_bend_factor: maxBendFactor,
        streamline_length: streamlineLength,
        step_size: stepSize,
        n_seeds: nSeeds,
        target_clusters: targetClusters,
        render_mode: renderMode
      };

      if (skip) {
        params.skip = parseInt(skip, 10);
      }

      console.log('[Velocity3DPanel] 开始生成3D可视化，参数:', params);
      const res = await fetch(`${API_BASE_URL}/api/velocity3d/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      console.log('[Velocity3DPanel] 收到后端响应，状态码:', res.status);

      const contentType = res.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await res.text();
        console.error('[Velocity3DPanel] Non-JSON response:', text.substring(0, 200));
        throw new Error(`后端返回了非JSON响应 (${res.status})。请检查后端服务器是否正常运行。`);
      }

      const data = await res.json();
      console.log('[Velocity3DPanel] 收到响应:', { 
        success: data.success, 
        hasImage: !!data.image, 
        imageLength: data.image ? data.image.length : 0,
        imagePrefix: data.image ? data.image.substring(0, 50) : 'null'
      });
      console.log('[Velocity3DPanel] 图像数据是否存在:', !!data.image);
      
      if (!res.ok || !data.success) {
        throw new Error(data.error || '生成3D可视化失败');
      }

      if (renderMode === 'image') {
        if (!data.image) {
          throw new Error('后端返回成功，但没有图像数据');
        }
        console.log('[Velocity3DPanel] 设置图像数据，长度:', data.image.length);
        setResultImage(data.image);
      } else {
        // 仅打开 PyVista 窗口，不返回截图
        setResultImage(null);
      }
      
      // 保存到历史记录
      if (onSaveHistory) {
        const methodName = `策略${strategyIdx} + 矢量场模式${vectorMode}`;
        onSaveHistory({
          method: methodName,
          timeStep: currentTimeStep || 'N/A',
          image: renderMode === 'image' ? data.image : undefined,
          timestamp: new Date().toISOString(),
          params: {
            strategy_idx: strategyIdx,
            vector_mode: vectorMode,
            lat_start: latStart,
            lat_end: latEnd,
            lon_start: lonStart,
            lon_end: lonEnd
          }
        });
      }
    } catch (err) {
      console.error('[Velocity3DPanel] Error generating visualization:', err);
      setError(err.message || '生成3D可视化失败');
      setResultImage(null);
    } finally {
      setLoading(false);
    }
  };

  // 条件返回必须在所有 hooks 之后
  if (!isVisible) return null;

  return (
    <div style={{
      position: 'absolute',
      top: '150px',
      left: '50%',
      transform: 'translateX(-50%)',
      width: '90%',
      maxWidth: 1400,
      backgroundColor: '#ffffff',
      borderRadius: 12,
      padding: 24,
      boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)',
      zIndex: 3500, // 提高 z-index，确保显示在最上层
      maxHeight: 'calc(100vh - 200px)',
      overflow: 'auto',
      display: 'block' // 确保显示
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ margin: 0 }}>3D可视化（策略+矢量场优化）</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => setShowConfig(!showConfig)} style={{
            backgroundColor: showConfig ? '#6b7280' : '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            padding: '8px 16px',
            cursor: 'pointer',
            fontSize: 14
          }}>
            {showConfig ? '隐藏参数' : '显示参数'}
          </button>
          <button onClick={onClose} style={{
            backgroundColor: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            padding: '8px 16px',
            cursor: 'pointer'
          }}>关闭</button>
        </div>
      </div>

      {showConfig && (
        <div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
          {/* 左侧：基础参数 */}
          <div>
            <h3 style={{ marginTop: 0, marginBottom: 12 }}>基础参数</h3>
            
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
              透明度策略 (1-20):
              <select value={strategyIdx} onChange={(e) => setStrategyIdx(parseInt(e.target.value, 10))} style={{
                width: '100%',
                padding: 8,
                borderRadius: 6,
                border: '1px solid #ccc',
                marginTop: 4
              }}>
                {strategies.map(s => (
                  <option key={s.id} value={s.id}>{s.id}: {s.description}</option>
                ))}
                {strategies.length === 0 && <option value={1}>1: 策略1（默认）</option>}
              </select>
            </label>

            <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
              矢量场模式:
              <select value={vectorMode} onChange={(e) => setVectorMode(parseInt(e.target.value, 10))} style={{
                width: '100%',
                padding: 8,
                borderRadius: 6,
                border: '1px solid #ccc',
                marginTop: 4
              }}>
                <option value={1}>模式1: 弯曲箭头</option>
                <option value={2}>模式2: 三维流线</option>
                <option value={3}>模式3: 聚类区域大箭头</option>
              </select>
            </label>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
              <label style={{ display: 'block', fontWeight: 600 }}>
                纬度起始: <input type="number" value={latStart} onChange={(e) => setLatStart(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
              <label style={{ display: 'block', fontWeight: 600 }}>
                纬度结束: <input type="number" value={latEnd} onChange={(e) => setLatEnd(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
              <label style={{ display: 'block', fontWeight: 600 }}>
                经度起始: <input type="number" value={lonStart} onChange={(e) => setLonStart(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
              <label style={{ display: 'block', fontWeight: 600 }}>
                经度结束: <input type="number" value={lonEnd} onChange={(e) => setLonEnd(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
              <label style={{ display: 'block', fontWeight: 600 }}>
                深度层数: <input type="number" value={nz} onChange={(e) => setNz(parseInt(e.target.value, 10))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
              <label style={{ display: 'block', fontWeight: 600 }}>
                数据质量: <input type="number" value={dataQuality} onChange={(e) => setDataQuality(parseInt(e.target.value, 10))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
              <label style={{ display: 'block', fontWeight: 600 }}>
                XY缩放: <input type="number" value={scaleXy} onChange={(e) => setScaleXy(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
              <label style={{ display: 'block', fontWeight: 600 }}>
                采样间隔: <input type="number" value={skip} onChange={(e) => setSkip(e.target.value)} placeholder="留空=自动" style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
            </div>

            {/* 返回方式 */}
            <div style={{ marginTop: 12 }}>
              <h4 style={{ margin: '8px 0', fontWeight: 600 }}>返回方式</h4>
              <div style={{ display: 'flex', gap: 12 }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    name="renderModeVelocity3D"
                    value="image"
                    checked={renderMode === 'image'}
                    onChange={() => setRenderMode('image')}
                    style={{ marginRight: 6, cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: '14px', color: '#374151' }}>返回截图（写入历史记录）</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    name="renderModeVelocity3D"
                    value="window"
                    checked={renderMode === 'window'}
                    onChange={() => setRenderMode('window')}
                    style={{ marginRight: 6, cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: '14px', color: '#374151' }}>直接打开 PyVista 窗口（不返回图）</span>
                </label>
              </div>
            </div>
          </div>

          {/* 右侧：矢量场参数 */}
          <div>
            <h3 style={{ marginTop: 0, marginBottom: 12 }}>矢量场参数</h3>

            {vectorMode === 1 && (
              <>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                  箭头缩放: <input type="number" step="0.1" value={arrowScale} onChange={(e) => setArrowScale(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
                </label>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                  邻域点数: <input type="number" value={kNeighbors} onChange={(e) => setKNeighbors(parseInt(e.target.value, 10))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
                </label>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                  最大弯曲因子: <input type="number" step="0.1" value={maxBendFactor} onChange={(e) => setMaxBendFactor(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
                </label>
              </>
            )}

            {vectorMode === 2 && (
              <>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                  流线长度: <input type="number" step="0.1" value={streamlineLength} onChange={(e) => setStreamlineLength(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
                </label>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                  积分步长: <input type="number" step="0.1" value={stepSize} onChange={(e) => setStepSize(parseFloat(e.target.value))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
                </label>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                  种子点数: <input type="number" value={nSeeds} onChange={(e) => setNSeeds(parseInt(e.target.value, 10))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
                </label>
              </>
            )}

            {vectorMode === 3 && (
              <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                目标聚类数: <input type="number" value={targetClusters} onChange={(e) => setTargetClusters(parseInt(e.target.value, 10))} style={{ width: '100%', padding: 6, borderRadius: 4, border: '1px solid #ccc', marginTop: 4 }} />
              </label>
            )}
          </div>
        </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            style={{
              width: '100%',
              backgroundColor: loading ? '#9ca3af' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              padding: 12,
              fontSize: 16,
              fontWeight: 600,
              cursor: loading ? 'not-allowed' : 'pointer',
              marginBottom: 20
            }}
          >
            {loading ? '生成中...' : '生成3D可视化'}
          </button>
        </div>
      )}

      {error && (
        <div style={{
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          color: '#991b1b',
          padding: 12,
          borderRadius: 6,
          marginBottom: 20
        }}>
          {error}
        </div>
      )}

      {loading && (
        <div style={{
          textAlign: 'center',
          padding: '40px',
          color: '#6b7280',
          fontSize: '16px'
        }}>
          <div>正在生成3D可视化...</div>
          <div style={{ marginTop: 10, fontSize: '14px' }}>这可能需要一些时间，请耐心等待</div>
        </div>
      )}

      {resultImage && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1f2937' }}>可视化结果：</h3>
          <div style={{
            backgroundColor: '#f9fafb',
            padding: '12px',
            borderRadius: 8,
            border: '3px solid #10b981'
          }}>
            <img
              src={resultImage}
              alt="3D Visualization"
              style={{
                width: '100%',
                height: 'auto',
                borderRadius: 8,
                border: '1px solid #ccc',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                display: 'block'
              }}
              onError={(e) => {
                console.error('[Velocity3DPanel] 图像加载失败:', e);
                console.error('[Velocity3DPanel] 图像数据前缀:', resultImage ? resultImage.substring(0, 100) : 'null');
                setError('图像加载失败，请检查图像数据格式。图像数据前缀: ' + (resultImage ? resultImage.substring(0, 50) : 'null'));
              }}
              onLoad={(e) => {
                console.log('[Velocity3DPanel] 图像加载成功，尺寸:', e.target.naturalWidth, 'x', e.target.naturalHeight);
              }}
            />
          </div>
        </div>
      )}

      {!loading && !resultImage && !error && (
        <div style={{
          textAlign: 'center',
          padding: '40px',
          color: '#9ca3af',
          fontSize: '14px'
        }}>
          点击"生成3D可视化"按钮开始生成
        </div>
      )}
    </div>
  );
};

// 大气3D可视化组件
const Atmosphere3DPanel = ({ isVisible, onClose, onSaveHistory, typhoonId, currentTime, center }) => {
  const [lonMin, setLonMin] = useState('');
  const [lonMax, setLonMax] = useState('');
  const [latMin, setLatMin] = useState('');
  const [latMax, setLatMax] = useState('');
  const [timeStep, setTimeStep] = useState(0);
  const [layerMin, setLayerMin] = useState(0);
  const [layerMax, setLayerMax] = useState(50);
  const [dataQuality, setDataQuality] = useState(-6);
  const [scaleXy, setScaleXy] = useState(25);
  const [atmosphereNz, setAtmosphereNz] = useState(20);
  const [vectorMode, setVectorMode] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [resultMsg, setResultMsg] = useState('');
  const [resultImage, setResultImage] = useState(null);
  const [renderMode, setRenderMode] = useState('image'); // 'image' or 'window'
  const [showConfig, setShowConfig] = useState(true);

  // 根据台风中心位置设置默认经纬范围
  useEffect(() => {
    if (isVisible && center) {
      const defaultRange = 5;
      setLatMin((center.lat - defaultRange).toFixed(2));
      setLatMax((center.lat + defaultRange).toFixed(2));
      setLonMin((center.lng - defaultRange).toFixed(2));
      setLonMax((center.lng + defaultRange).toFixed(2));
      setTimeStep(currentTime - 1);
    }
  }, [isVisible, center, currentTime]);

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      setResultMsg('');
      setResultImage(null);

      if (!lonMin || !lonMax || !latMin || !latMax) {
        throw new Error('请填写所有经纬度范围参数');
      }

      const params = {
        lon_min: parseFloat(lonMin),
        lon_max: parseFloat(lonMax),
        lat_min: parseFloat(latMin),
        lat_max: parseFloat(latMax),
        time_step: parseInt(timeStep, 10),
        layer_min: parseInt(layerMin, 10),
        layer_max: parseInt(layerMax, 10),
        data_quality: parseInt(dataQuality, 10),
        scale_xy: parseFloat(scaleXy),
        atmosphere_nz: parseInt(atmosphereNz, 10),
        vector_mode: vectorMode,
        render_mode: renderMode
      };

      console.log('[Atmosphere3DPanel] 开始生成大气3D可视化，参数:', params);
      
      const res = await fetch(`${API_BASE_URL}/api/atmosphere-3d/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || '生成大气3D可视化失败');
      }
      if (renderMode === 'image') {
        if (!data.image) throw new Error('后端未返回图像数据');
        const image = data.image || null;
        if (image) setResultImage(image);
      } else {
        setResultImage(null);
      }

      if (onSaveHistory) {
        onSaveHistory({
          type: '大气3D可视化',
          typhoonId: typhoonId,
          time: currentTime,
          params: params,
          timestamp: new Date().toISOString(),
          method: '大气3D可视化',
          image: renderMode === 'image' ? data.image : undefined
        });
      }

      setResultMsg(renderMode === 'image'
        ? '已生成结果，已写入历史记录，可在下方预览。'
        : '已在 PyVista 窗口启动（不返回截图）。');
      
    } catch (err) {
      console.error('[Atmosphere3DPanel] Error generating visualization:', err);
      setError(err.message || '生成大气3D可视化失败');
      setResultMsg('');
      setResultImage(null);
    } finally {
      setLoading(false);
    }
  };

  if (!isVisible) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 4000
    }}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: 12,
        padding: 24,
        width: '90%',
        maxWidth: 700,
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0, color: '#1f2937' }}>🌬️ 大气3D可视化</h2>
          <button
            onClick={onClose}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: 24,
              cursor: 'pointer',
              color: '#6b7280'
            }}
          >
            ×
          </button>
        </div>

        {error && (
          <div style={{
            padding: 12,
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 6,
            color: '#dc2626',
            marginBottom: 16
          }}>
            {error}
          </div>
        )}

        {resultMsg && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: '#ecfdf3',
            borderRadius: 8,
            border: '1px solid #bbf7d0',
            color: '#166534',
            fontSize: '14px'
          }}>
            ✅ {resultMsg}
          </div>
        )}

        {renderMode === 'image' && resultImage && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: '#f8fafc',
            borderRadius: 8,
            border: '1px solid #e5e7eb'
          }}>
            <div style={{ marginBottom: 8, fontWeight: 600, color: '#1f2937' }}>结果预览</div>
            <img
              src={`data:image/png;base64,${resultImage}`}
              alt="海气耦合结果"
              style={{ width: '100%', borderRadius: 6 }}
            />
          </div>
        )}

        {resultImage && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: '#f8fafc',
            borderRadius: 8,
            border: '1px solid #e5e7eb'
          }}>
            <div style={{ marginBottom: 8, fontWeight: 600, color: '#1f2937' }}>结果预览</div>
            <img src={`data:image/png;base64,${resultImage}`} alt="大气3D结果" style={{ width: '100%', borderRadius: 6 }} />
          </div>
        )}

        <div style={{ marginBottom: 20 }}>
          <button
            onClick={() => setShowConfig(!showConfig)}
            style={{
              backgroundColor: showConfig ? '#6b7280' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              padding: '8px 16px',
              fontSize: '14px',
              cursor: 'pointer',
              marginBottom: 16
            }}
          >
            {showConfig ? '隐藏参数配置' : '显示参数配置'}
          </button>

          {showConfig && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* 经纬范围 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>经纬范围</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    经度最小值 (lon_min)
                    <input
                      type="number"
                      value={lonMin}
                      onChange={(e) => setLonMin(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    经度最大值 (lon_max)
                    <input
                      type="number"
                      value={lonMax}
                      onChange={(e) => setLonMax(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    纬度最小值 (lat_min)
                    <input
                      type="number"
                      value={latMin}
                      onChange={(e) => setLatMin(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    纬度最大值 (lat_max)
                    <input
                      type="number"
                      value={latMax}
                      onChange={(e) => setLatMax(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                </div>
              </div>

              {/* 时间步和层数 */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                  时间步 (time_step)
                  <input
                    type="number"
                    value={timeStep}
                    onChange={(e) => setTimeStep(e.target.value)}
                    style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                    min="0"
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                  大气层数 (atmosphere_nz)
                  <input
                    type="number"
                    value={atmosphereNz}
                    onChange={(e) => setAtmosphereNz(e.target.value)}
                    style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                    min="2"
                    max="50"
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                  层数最小值 (layer_min)
                  <input
                    type="number"
                    value={layerMin}
                    onChange={(e) => setLayerMin(e.target.value)}
                    style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                    min="0"
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                  层数最大值 (layer_max)
                  <input
                    type="number"
                    value={layerMax}
                    onChange={(e) => setLayerMax(e.target.value)}
                    style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                    min="1"
                    max="50"
                  />
                </label>
              </div>

              {/* 高级参数 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>高级参数</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    数据质量 (data_quality)
                    <input
                      type="number"
                      value={dataQuality}
                      onChange={(e) => setDataQuality(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      min="-10"
                      max="0"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    XY缩放 (scale_xy)
                    <input
                      type="number"
                      value={scaleXy}
                      onChange={(e) => setScaleXy(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      min="1"
                      step="1"
                    />
                  </label>
                </div>
              </div>

                {/* 矢量场模式 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>矢量场模式</h3>
                <div style={{ display: 'flex', gap: 12 }}>
                  {[1, 2, 3].map((mode) => (
                    <label key={mode} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="radio"
                        name="vectorMode"
                        value={mode}
                        checked={vectorMode === mode}
                        onChange={(e) => setVectorMode(parseInt(e.target.value, 10))}
                        style={{ marginRight: 6, cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '14px', color: '#374151' }}>
                        {mode === 1 ? '弯曲箭头' : mode === 2 ? '三维流线' : '直线箭头'}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

                {/* 渲染方式 */}
                <div>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                    返回方式
                  </h3>
                  <div style={{ display: 'flex', gap: 12 }}>
                    <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="radio"
                        name="renderModeA3D"
                        value="image"
                        checked={renderMode === 'image'}
                        onChange={() => setRenderMode('image')}
                        style={{ marginRight: 6, cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '14px', color: '#374151' }}>返回截图</span>
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="radio"
                        name="renderModeA3D"
                        value="window"
                        checked={renderMode === 'window'}
                        onChange={() => setRenderMode('window')}
                        style={{ marginRight: 6, cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '14px', color: '#374151' }}>直接打开 PyVista 窗口（不返回图）</span>
                    </label>
                  </div>
                </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', marginTop: 24, paddingTop: 20, borderTop: '1px solid #e5e7eb' }}>
          <button
            onClick={onClose}
            style={{
              padding: '10px 20px',
              backgroundColor: '#e5e7eb',
              color: '#374151',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 500
            }}
          >
            取消
          </button>
          <button
            onClick={handleGenerate}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: loading ? '#94a3b8' : '#10b981',
              color: '#ffffff',
              border: 'none',
              borderRadius: 6,
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 500
            }}
          >
            {loading ? '生成中...' : '生成大气3D可视化'}
          </button>
        </div>

        {loading && (
          <div style={{
            marginTop: 20,
            padding: 20,
            background: '#f0f9ff',
            borderRadius: 8,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '16px', color: '#0369a1', marginBottom: 10 }}>⏳ 正在生成大气3D可视化...</div>
            <div style={{ marginTop: 10, fontSize: '14px', color: '#64748b' }}>
              这可能需要一些时间，请耐心等待。可视化将在PyVista窗口中显示。
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// 海气耦合组件
const AtmosphereOceanFusionPanel = ({ isVisible, onClose, onSaveHistory, typhoonId, currentTime, center }) => {
  const [lonMin, setLonMin] = useState('');
  const [lonMax, setLonMax] = useState('');
  const [latMin, setLatMin] = useState('');
  const [latMax, setLatMax] = useState('');
  const [timeStep, setTimeStep] = useState(0);
  const [resolution, setResolution] = useState('medium'); // 保留但后端忽略
  const [vectorMode, setVectorMode] = useState(3); // 默认使用直线箭头（贴合立方体）
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [resultMsg, setResultMsg] = useState('');
  const [resultImage, setResultImage] = useState(null);
  const [renderMode, setRenderMode] = useState('image'); // 'image' 或 'window'
  const [showConfig, setShowConfig] = useState(true);

  // 根据台风中心位置和ID设置默认经纬范围
  useEffect(() => {
    if (isVisible && center) {
      // 根据台风位置设置默认范围（±5度）
      const defaultRange = 5;
      setLatMin((center.lat - defaultRange).toFixed(2));
      setLatMax((center.lat + defaultRange).toFixed(2));
      setLonMin((center.lng - defaultRange).toFixed(2));
      setLonMax((center.lng + defaultRange).toFixed(2));
      setTimeStep(currentTime - 1); // 转换为0-based索引
    }
  }, [isVisible, center, currentTime]);

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      setResultMsg('');

      // 验证参数
      if (!lonMin || !lonMax || !latMin || !latMax) {
        throw new Error('请填写所有经纬度范围参数');
      }

      const params = {
        lon_min: parseFloat(lonMin),
        lon_max: parseFloat(lonMax),
        lat_min: parseFloat(latMin),
        lat_max: parseFloat(latMax),
        time_step: parseInt(timeStep, 10),
        // 新的贴合立方体参数（后端有默认值，这里显式传递以便调整）
        layer_min: 0,
        layer_max: 50,
        ocean_nz: 40,
        atmosphere_nz: 20,
        data_quality: -6,
        scale_xy: 25,
        vector_mode: vectorMode,
        render_mode: renderMode
      };

      console.log('[AtmosphereOceanFusionPanel] 开始生成上下贴合海气立方体可视化，参数:', params);
      
      const res = await fetch(`${API_BASE_URL}/api/atmosphere-ocean-coupled/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || '生成海气耦合可视化失败');
      }

      if (renderMode === 'image') {
        if (!data.image) {
          throw new Error('后端未返回图像数据');
        }
        setResultImage(data.image || null);
      } else {
        // 仅打开 PyVista 窗口，不返回截图
        setResultImage(null);
      }

      // 保存到历史记录
      if (onSaveHistory) {
        onSaveHistory({
          type: '海气耦合',
          typhoonId: typhoonId,
          time: currentTime,
          params: params,
          method: '上下贴合海气立方体',
          timestamp: new Date().toISOString(),
          image: renderMode === 'image' ? data.image : undefined
        });
      }

      setResultMsg(
        renderMode === 'image'
          ? '已生成结果，已写入历史记录，可在下方预览。'
          : '已在 PyVista 窗口启动（不返回截图）。'
      );
      
    } catch (err) {
      console.error('[AtmosphereOceanFusionPanel] Error generating visualization:', err);
      setError(err.message || '生成海气耦合可视化失败');
      setResultMsg('');
      setResultImage(null);
    } finally {
      setLoading(false);
    }
  };

  if (!isVisible) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 4000
    }}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: 12,
        padding: 24,
        width: '90%',
        maxWidth: 600,
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0, color: '#1f2937' }}>🌊 海气耦合可视化</h2>
          <button
            onClick={onClose}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: 24,
              cursor: 'pointer',
              color: '#6b7280',
              padding: '4px 8px'
            }}
          >
            ×
          </button>
        </div>

        {error && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: '#fef2f2',
            borderRadius: 8,
            border: '1px solid #fecaca',
            color: '#dc2626',
            fontSize: '14px'
          }}>
            ❌ {error}
          </div>
        )}

        {resultMsg && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: '#ecfdf3',
            borderRadius: 8,
            border: '1px solid #bbf7d0',
            color: '#166534',
            fontSize: '14px'
          }}>
            ✅ {resultMsg}
          </div>
        )}

        {renderMode === 'image' && resultImage && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: '#f8fafc',
            borderRadius: 8,
            border: '1px solid #e5e7eb'
          }}>
            <div style={{ marginBottom: 8, fontWeight: 600, color: '#1f2937' }}>结果预览</div>
            <img
              src={`data:image/png;base64,${resultImage}`}
              alt="海气耦合结果"
              style={{ width: '100%', borderRadius: 6 }}
            />
          </div>
        )}

        <div style={{ marginBottom: 20 }}>
          <button
            onClick={() => setShowConfig(!showConfig)}
            style={{
              backgroundColor: showConfig ? '#6b7280' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              padding: '8px 16px',
              fontSize: '14px',
              cursor: 'pointer',
              marginBottom: 16
            }}
          >
            {showConfig ? '隐藏参数配置' : '显示参数配置'}
          </button>

          {showConfig && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* 经纬范围 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                  经纬范围
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    左下角经度 (lon_min)
                    <input
                      type="number"
                      value={lonMin}
                      onChange={(e) => setLonMin(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    左下角纬度 (lat_min)
                    <input
                      type="number"
                      value={latMin}
                      onChange={(e) => setLatMin(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    右上角经度 (lon_max)
                    <input
                      type="number"
                      value={lonMax}
                      onChange={(e) => setLonMax(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                    右上角纬度 (lat_max)
                    <input
                      type="number"
                      value={latMax}
                      onChange={(e) => setLatMax(e.target.value)}
                      style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                      step="0.1"
                    />
                  </label>
                </div>
              </div>

              {/* 时间步 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                  时间步
                </h3>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '13px' }}>
                  时间步索引 (time_step)
                  <input
                    type="number"
                    value={timeStep}
                    onChange={(e) => setTimeStep(e.target.value)}
                    style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                    min="0"
                  />
                </label>
              </div>

              {/* 分辨率 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                  分辨率
                </h3>
                <div style={{ display: 'flex', gap: 12 }}>
                  {['low', 'medium', 'high'].map((res) => (
                    <label key={res} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="radio"
                        name="resolution"
                        value={res}
                        checked={resolution === res}
                        onChange={(e) => setResolution(e.target.value)}
                        style={{ marginRight: 6, cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '14px', color: '#374151' }}>
                        {res === 'low' ? '低' : res === 'medium' ? '中' : '高'}
                      </span>
                    </label>
                  ))}
                </div>
                <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#6b7280' }}>
                  {resolution === 'low' && '采样间隔4，海洋5层，大气5层，箭头200个'}
                  {resolution === 'medium' && '采样间隔2，海洋10层，大气10层，箭头500个'}
                  {resolution === 'high' && '采样间隔1，海洋20层，大气20层，箭头1000个'}
                </p>
              </div>

              {/* 矢量场模式 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                  矢量场模式
                </h3>
                <div style={{ display: 'flex', gap: 12 }}>
                  {[1, 2, 3].map((mode) => (
                    <label key={mode} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="radio"
                        name="vectorModeAOF"
                        value={mode}
                        checked={vectorMode === mode}
                        onChange={(e) => setVectorMode(parseInt(e.target.value, 10))}
                        style={{ marginRight: 6, cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '14px', color: '#374151' }}>
                        {mode === 1 ? '模式1 - 弯曲箭头' : mode === 2 ? '模式2 - 三维流线' : '模式3 - 直线箭头（贴合立方体）'}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* 返回方式 */}
              <div>
                <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                  返回方式
                </h3>
                <div style={{ display: 'flex', gap: 12 }}>
                  <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="radio"
                      name="renderModeAOF"
                      value="image"
                      checked={renderMode === 'image'}
                      onChange={() => setRenderMode('image')}
                      style={{ marginRight: 6, cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '14px', color: '#374151' }}>返回截图</span>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="radio"
                      name="renderModeAOF"
                      value="window"
                      checked={renderMode === 'window'}
                      onChange={() => setRenderMode('window')}
                      style={{ marginRight: 6, cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '14px', color: '#374151' }}>直接打开 PyVista 窗口（不返回图）</span>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', marginTop: 24, paddingTop: 20, borderTop: '1px solid #e5e7eb' }}>
          <button
            onClick={onClose}
            style={{
              padding: '10px 20px',
              backgroundColor: '#e5e7eb',
              color: '#374151',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 500
            }}
          >
            取消
          </button>
          <button
            onClick={handleGenerate}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: loading ? '#94a3b8' : '#06b6d4',
              color: '#ffffff',
              border: 'none',
              borderRadius: 6,
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 500
            }}
          >
            {loading ? '生成中...' : '生成海气耦合可视化'}
          </button>
        </div>

        {loading && (
          <div style={{
            marginTop: 20,
            padding: 20,
            background: '#f0f9ff',
            borderRadius: 8,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '16px', color: '#0369a1', marginBottom: 10 }}>⏳ 正在生成海气耦合可视化...</div>
            <div style={{ marginTop: 10, fontSize: '14px', color: '#64748b' }}>
              这可能需要一些时间，请耐心等待。可视化将在PyVista窗口中显示。
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// 截面功能组件
const CrossSectionPanel = ({ typhoonId, isVisible, onClose, onSaveHistory, currentTimeStep }) => {
  const [method, setMethod] = useState('three_points'); // 'three_points' or 'view_line'
  const [params, setParams] = useState({
    p1: [0, 0, 0],
    p2: [100, 0, 0],
    p3: [0, 100, 0],
    view_direction: [1, 0, 0],
    depth_offset: 0
  });
  const [loading, setLoading] = useState(false);
  const [resultImage, setResultImage] = useState(null);
  const [error, setError] = useState(null);

  // 条件返回必须在所有 hooks 之后
  if (!isVisible) return null;

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const requestParams = method === 'three_points' 
        ? { p1: params.p1, p2: params.p2, p3: params.p3 }
        : { view_direction: params.view_direction, depth_offset: params.depth_offset };
      
      const res = await fetch(`${API_BASE_URL}/api/typhoon/${typhoonId}/cross-section`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method,
          params: requestParams,
          resolution: 150
        })
      });
      
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || '生成截面失败');
      }
      
      setResultImage(data.image);
      
      // 保存到历史记录
      if (onSaveHistory) {
        const methodName = method === 'three_points' ? '三点定义平面' : '视图方向+深度偏移';
        onSaveHistory({
          method: `截面分析 - ${methodName}`,
          timeStep: currentTimeStep || 'N/A',
          image: data.image,
          timestamp: new Date().toISOString(),
          params: {
            method: method,
            ...requestParams
          }
        });
      }
    } catch (err) {
      setError(err.message || '生成截面失败');
      setResultImage(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      zIndex: 4000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 20
    }}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: 12,
        padding: 24,
        maxWidth: 800,
        maxHeight: '90vh',
        overflow: 'auto',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0 }}>海洋截面可视化</h2>
          <button onClick={onClose} style={{
            backgroundColor: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            padding: '8px 16px',
            cursor: 'pointer'
          }}>关闭</button>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>选择方法：</label>
          <select value={method} onChange={(e) => setMethod(e.target.value)} style={{
            width: '100%',
            padding: 8,
            borderRadius: 6,
            border: '1px solid #ccc'
          }}>
            <option value="three_points">方法1：三个点定义平面</option>
            <option value="view_line">方法2：视图方向+深度偏移</option>
          </select>
        </div>

        {method === 'three_points' ? (
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>三个点坐标 (x, y, z)：</label>
            {['p1', 'p2', 'p3'].map((key, idx) => (
              <div key={key} style={{ marginBottom: 12 }}>
                <label style={{ display: 'block', marginBottom: 4 }}>点{idx + 1}：</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  {['x', 'y', 'z'].map((coord, i) => (
                    <input
                      key={coord}
                      type="number"
                      value={params[key][i]}
                      onChange={(e) => {
                        const newParams = { ...params };
                        newParams[key] = [...newParams[key]];
                        newParams[key][i] = parseFloat(e.target.value) || 0;
                        setParams(newParams);
                      }}
                      placeholder={coord}
                      style={{
                        flex: 1,
                        padding: 8,
                        borderRadius: 6,
                        border: '1px solid #ccc'
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>视图方向向量 (x, y, z)：</label>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              {['x', 'y', 'z'].map((coord, i) => (
                <input
                  key={coord}
                  type="number"
                  value={params.view_direction[i]}
                  onChange={(e) => {
                    const newParams = { ...params };
                    newParams.view_direction = [...newParams.view_direction];
                    newParams.view_direction[i] = parseFloat(e.target.value) || 0;
                    setParams(newParams);
                  }}
                  placeholder={coord}
                  style={{
                    flex: 1,
                    padding: 8,
                    borderRadius: 6,
                    border: '1px solid #ccc'
                  }}
                />
              ))}
            </div>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>深度偏移：</label>
            <input
              type="number"
              value={params.depth_offset}
              onChange={(e) => setParams({ ...params, depth_offset: parseFloat(e.target.value) || 0 })}
              style={{
                width: '100%',
                padding: 8,
                borderRadius: 6,
                border: '1px solid #ccc'
              }}
            />
          </div>
        )}

        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            width: '100%',
            backgroundColor: loading ? '#ccc' : '#6366f1',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            padding: '12px',
            fontSize: 16,
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            marginBottom: 20
          }}
        >
          {loading ? '生成中...' : '生成截面'}
        </button>

        {error && (
          <div style={{
            backgroundColor: '#fee2e2',
            color: '#dc2626',
            padding: 12,
            borderRadius: 6,
            marginBottom: 20
          }}>
            {error}
          </div>
        )}

        {resultImage && (
          <div>
            <h3 style={{ marginBottom: 12 }}>截面结果：</h3>
            <img
              src={resultImage}
              alt="Cross-section"
              style={{
                width: '100%',
                borderRadius: 8,
                border: '1px solid #ccc'
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export const TyphoonPage = ({ onBack, typhoonId, currentTime, onTimeChange, currentHeight, onHeightChange, open3D=false, useSimulation=true, onSaveHistory, initialOpen3DCube=false, initialOpenCrossSection=false, initialOpenVelocity3D=false }) => {
  const [show3DCube, setShow3DCube] = useState(initialOpen3DCube);
  const [showSaltConfig, setShowSaltConfig] = useState(false);
  const [showCrossSection, setShowCrossSection] = useState(initialOpenCrossSection);
  const [showVelocity3D, setShowVelocity3D] = useState(initialOpenVelocity3D);
  const [showAtmosphereOceanFusion, setShowAtmosphereOceanFusion] = useState(false);
  const [showAtmosphere3D, setShowAtmosphere3D] = useState(false);
  const [center, setCenter] = useState(null); // { lat, lng }

  // 根据台风ID获取台风信息
  const getTyphoonInfo = (id) => {
    if (id === 1) {
      return {
        name: '台风1',
        location: '南海区域',
        position: '南海'
      };
    } else if (id === 2) {
      return {
        name: '台风2',
        location: '西太平洋区域',
        position: '西太平洋'
      };
    } else if (id === 3) {
      return {
        name: '台风3',
        location: '青岛近海（胶州湾-黄海交界）',
        position: '青岛近海',
        note: '3D视图直接拉起 PyVista 场景，展示 text.py 产生的盐度层 + 流线效果'
      };
    } else {
      // 支持动态台风ID（从追踪界面来的）
      return {
        name: `台风${id}`,
        location: '检测到的台风',
        position: '追踪位置'
      };
    }
  };

  const typhoonInfo = getTyphoonInfo(typhoonId);
  const isTyphoon3 = typhoonId === 3;
  
  // 不进行 early return ，确保 Hooks 在每次渲染中保持调用顺序一致

  // 当typhoonId或时间改变时，从后端获取台风中心位置
  useEffect(() => {
    let ignore = false;
    const fetchTyPos = async () => {
      try {
        const timeIndex = Math.max(0, currentTime - 1);
        if (useSimulation) {
          const baseLookup = {
            1: { lat: 17.5, lng: 115 },
            2: { lat: 25, lng: 140 },
            3: { lat: 36.05, lng: 120.35 }
          };
          const base = baseLookup[typhoonId] || { lat: 20, lng: 120 };
          const jitter = (timeIndex % 3) * 0.2; // slight variation
          setCenter({ lat: base.lat + jitter, lng: base.lng - jitter, grid_index: { ix: 0, iy: 0 } });
        } else {
          const res = await fetch(`${API_BASE_URL}/api/typhoon?time=${timeIndex}&id=${typhoonId}`);
          if (!res.ok) throw new Error(`Status ${res.status}`);
          const data = await res.json();
          if (data && data.success && !ignore) {
            setCenter({ lat: data.lat, lng: data.lng, grid_index: data.grid_index });
          }
        }
      } catch (err) {
        console.warn('Failed to get typhoon location from backend:', err);
        setCenter(null);
      }
    };
    fetchTyPos();
    return () => { ignore = true; };
  }, [typhoonId, currentTime]);

  // 接收open3D请求，默认打开3D（并重置flag）
  useEffect(() => {
    if (open3D) {
      setShow3DCube(true);
    }
  }, [open3D]);

  // 处理初始打开状态
  useEffect(() => {
    if (initialOpen3DCube) {
      setShow3DCube(true);
    }
    if (initialOpenCrossSection) {
      setShowCrossSection(true);
    }
    if (initialOpenVelocity3D) {
      setShowVelocity3D(true);
    }
  }, []); // 只在组件挂载时执行一次

  // 在 Hooks 之后再根据 typhoonInfo 做条件渲染，避免在 Hooks 之前早退导致 Hook 顺序不一致
  if (!typhoonInfo) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#fff'
      }}>
        <h2>未选择台风或台风信息不可用</h2>
        <button
          style={{ backgroundColor: '#4299e1', color: 'white', padding: '8px 12px', borderRadius: 6 }}
          onClick={onBack}
        >返回地图</button>
      </div>
    );
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: '#ffffff',
      zIndex: 3000,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      boxSizing: 'border-box'
    }}>
      {/* 返回地图按钮 */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          backgroundColor: '#4299e1',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '10px 20px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          zIndex: 2001,
          boxShadow: '0 2px 8px rgba(66, 153, 225, 0.3)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}
        onMouseOver={(e) => {
          e.target.style.backgroundColor = '#3182ce';
          e.target.style.boxShadow = '0 4px 12px rgba(66, 153, 225, 0.5)';
        }}
        onMouseOut={(e) => {
          e.target.style.backgroundColor = '#4299e1';
          e.target.style.boxShadow = '0 2px 8px rgba(66, 153, 225, 0.3)';
        }}
      >
        <span>←</span>
        <span>返回地图</span>
      </button>

      {/* 按钮组 */}
      <div style={{
          position: 'absolute',
          bottom: '100px',
          left: '50%',
          transform: 'translateX(-50%)',
        display: 'flex',
        gap: '12px',
        zIndex: 2001
      }}>
        {/* 盐度3D可视化按钮（运行 text.py，使用区域参数） */}
        <button
          onClick={() => setShowSaltConfig(true)}
          style={{
          backgroundColor: '#8b5cf6',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '12px 24px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)'
        }}
        onMouseOver={(e) => e.target.style.backgroundColor = '#7c3aed'}
        onMouseOut={(e) => e.target.style.backgroundColor = '#8b5cf6'}
      >
        {isTyphoon3 ? '运行 text.py 3D视图' : '盐度3D可视化'}
      </button>

        {/* 取截面按钮 */}
        <button
          onClick={() => setShowCrossSection(true)}
          style={{
            backgroundColor: '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '12px 24px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#059669'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#10b981'}
        >
          取截面
        </button>

        {/* 3D可视化按钮 */}
        <button
          onClick={() => setShowVelocity3D(true)}
          style={{
            backgroundColor: '#f59e0b',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '12px 24px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 12px rgba(245, 158, 11, 0.3)'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#d97706'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#f59e0b'}
        >
          海洋3D可视化
        </button>

        {/* 海气耦合按钮 */}
        <button
          onClick={() => setShowAtmosphereOceanFusion(true)}
          style={{
            backgroundColor: '#06b6d4',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '12px 24px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 12px rgba(6, 182, 212, 0.3)'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#0891b2'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#06b6d4'}
        >
          海气耦合
        </button>

        {/* 大气3D可视化按钮 */}
        <button
          onClick={() => setShowAtmosphere3D(true)}
          style={{
            backgroundColor: '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '12px 24px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#059669'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#10b981'}
        >
          大气3D可视化
        </button>

      </div>

      {/* 经纬度网格和3D数据 */}
      {!show3DCube && !isTyphoon3 && (
        <LatLngGrid 
          typhoonId={typhoonId} 
          currentTime={currentTime}
          currentHeight={currentHeight}
        />
      )}
      {!show3DCube && isTyphoon3 && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '70%',
          maxWidth: '900px',
          padding: '32px',
          borderRadius: '16px',
          background: 'linear-gradient(135deg, rgba(30,64,175,0.12), rgba(15,118,110,0.12))',
          border: '1px dashed rgba(59,130,246,0.5)',
          textAlign: 'center',
          color: '#0f172a'
        }}>
          <h3 style={{ marginTop: 0, marginBottom: 12 }}>台风3 · PyVista 专用 3D 场景</h3>
          <p style={{ margin: '8px 0', color: '#1e293b' }}>
            点击下方“运行 text.py 3D视图”按钮将实时调用 `src/text.py`，按照 `backend/config.json` 中的局部区域参数生成离屏渲染截图。
          </p>
          <p style={{ margin: '8px 0', color: '#334155' }}>
            建议先在前端配置面板调整区域范围，再打开 3D 视图，以便通过 PyVista 查看青岛附近最新的盐度层和流线结构。
          </p>
          <p style={{ margin: '8px 0', color: '#475569' }}>
            渲染完成后可在 3D 模态框中刷新或下载截图，支持与线下 `text.py` 运行结果对比。
          </p>
        </div>
      )}

      {/* 盐度3D可视化区域参数配置 + 触发 text.py */}
      <Salt3DConfigPanel
        visible={showSaltConfig}
        onClose={() => setShowSaltConfig(false)}
        onApply={() => {
          // 配置保存成功后，再弹出 PyVista 截图查看器
          setShow3DCube(true);
        }}
      />

      {/* PyVista 图像查看器（所有台风都使用，读取 text.py 截图） */}
      <Typhoon3Viewer
        isVisible={show3DCube}
        onClose={() => setShow3DCube(false)}
        typhoonId={typhoonId}
      />

      {/* 3D可视化面板 */}
      <Velocity3DPanel
        isVisible={showVelocity3D}
        onClose={() => setShowVelocity3D(false)}
        onSaveHistory={onSaveHistory}
        currentTimeStep={currentTime}
      />

      {/* 海气耦合面板 */}
      <AtmosphereOceanFusionPanel
        isVisible={showAtmosphereOceanFusion}
        onClose={() => setShowAtmosphereOceanFusion(false)}
        onSaveHistory={onSaveHistory}
        typhoonId={typhoonId}
        currentTime={currentTime}
        center={center}
      />

      {/* 大气3D可视化面板 */}
      <Atmosphere3DPanel
        isVisible={showAtmosphere3D}
        onClose={() => setShowAtmosphere3D(false)}
        onSaveHistory={onSaveHistory}
        typhoonId={typhoonId}
        currentTime={currentTime}
        center={center}
      />

      {/* 截面功能面板 */}
      <CrossSectionPanel
        typhoonId={typhoonId}
        isVisible={showCrossSection}
        onClose={() => setShowCrossSection(false)}
        onSaveHistory={onSaveHistory}
        currentTimeStep={currentTime}
      />

      <h1 style={{ color: '#2d3748', marginBottom: '30px', zIndex: 2002, position: 'relative' }}>{typhoonInfo.name} 详细信息</h1>
      
      {/* 台风信息 */}
      {typhoonId === 2 && currentTime === 1 ? (
        // 台风2在时间1下无数据
        <div style={{
          backgroundColor: '#fef2f2',
          border: '2px solid #fecaca',
          padding: '20px',
          borderRadius: '8px',
          width: '80%',
          maxWidth: '600px',
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          <h3 style={{ color: '#dc2626', margin: '0 0 16px 0' }}>⚠️ 时间1下无数据</h3>
          <p style={{ margin: '8px 0', fontSize: '14px', color: '#991b1b' }}>
            {typhoonInfo.name}在时间1下没有相关信息
          </p>
          <p style={{ margin: '8px 0', fontSize: '14px', color: '#991b1b' }}>
            请切换到时间2查看{typhoonInfo.name}的详细信息
          </p>
        </div>
      ) : (
        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '20px',
          borderRadius: '8px',
          width: '80%',
          maxWidth: '600px',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#4a5568', margin: '0 0 16px 0' }}>{typhoonInfo.name}（{typhoonInfo.position}）</h3>
          <p style={{ margin: '8px 0', fontSize: '14px', color: '#2d3748' }}>
            当前时间：时间{currentTime}
          </p>
          <p style={{ margin: '8px 0', fontSize: '14px', color: '#2d3748' }}>
            当前高度：高度{currentHeight}
          </p>
          <p style={{ margin: '8px 0', fontSize: '14px', color: '#2d3748' }}>
            位置：{typhoonInfo.location}
          </p>
          {center && (
            <p style={{ margin: '8px 0', fontSize: '14px', color: '#2d3748' }}>
              中心：{center.lat.toFixed(4)}, {center.lng.toFixed(4)}
            </p>
          )}
          <p style={{ margin: '8px 0', fontSize: '14px', color: '#2d3748' }}>
            时间{currentTime} - 高度{currentHeight}状态：这里是{typhoonInfo.name}在时间{currentTime}、高度{currentHeight}的详细信息
          </p>
          {typhoonInfo.note && (
            <p style={{ margin: '8px 0', fontSize: '13px', color: '#6366f1', fontWeight: 600 }}>
              {typhoonInfo.note}
            </p>
          )}
        </div>
      )}

      <button
        style={{
          backgroundColor: '#4299e1',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '12px 30px',
          fontSize: '1rem',
          cursor: 'pointer',
          transition: 'background-color 0.3s ease'
        }}
        onMouseOver={(e) => e.target.style.backgroundColor = '#3182ce'}
        onMouseOut={(e) => e.target.style.backgroundColor = '#4299e1'}
        onClick={onBack}
      >
        返回地图
      </button>
    </div>
  );
};

