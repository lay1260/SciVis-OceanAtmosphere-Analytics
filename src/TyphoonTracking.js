import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Circle, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// 修复 Leaflet 图标路径问题
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// 台风标记图标（增大尺寸）
const createTyphoonIcon = (typhoonId, isActive) => {
  return L.divIcon({
    className: 'typhoon-tracking-marker',
    html: `
      <div style="
        width: 40px;
        height: 40px;
        background-color: ${isActive ? '#dc2626' : '#94a3b8'};
        border: 4px solid white;
        border-radius: 50%;
        box-shadow: 0 3px 12px rgba(0,0,0,0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 16px;
      ">
        ${typhoonId}
      </div>
    `,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
  });
};

// 地图自动定位组件
const MapAutoCenter = ({ positions }) => {
  const map = useMap();
  
  useEffect(() => {
    if (positions && Object.keys(positions).length > 0) {
      // 计算所有台风中心的平均位置
      const lats = [];
      const lngs = [];
      
      Object.values(positions).forEach(pos => {
        if (pos && pos.lat && pos.lng) {
          lats.push(pos.lat);
          lngs.push(pos.lng);
        }
      });
      
      if (lats.length > 0 && lngs.length > 0) {
        const centerLat = lats.reduce((a, b) => a + b, 0) / lats.length;
        const centerLng = lngs.reduce((a, b) => a + b, 0) / lngs.length;
        
        // 计算合适的缩放级别（根据台风数量调整）
        const zoom = Object.keys(positions).length === 1 ? 7 : 6;
        
        // 平滑定位到中心
        map.flyTo([centerLat, centerLng], zoom, {
          duration: 1.5,
          easeLinearity: 0.25
        });
      }
    }
  }, [positions, map]);
  
  return null;
};

const TyphoonTracking = ({ onBack, onTyphoonClick, isMinimized = false, onRestore }) => {
  const [tracks, setTracks] = useState({});
  const [currentTimeStep, setCurrentTimeStep] = useState(4320);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trackingProgress, setTrackingProgress] = useState(null);
  const [selectedTyphoon, setSelectedTyphoon] = useState(null);
  const [trackingMode, setTrackingMode] = useState('all'); // 'all' 或 'single'
  const [timeMeta, setTimeMeta] = useState({ timesteps: [], base_time: null, step_hours: null });

  // 加载时间元数据（时间步 → 实际时间）
  useEffect(() => {
    const loadTimeMeta = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/time/metadata`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data && data.success) {
          setTimeMeta({
            timesteps: Array.isArray(data.timesteps) ? data.timesteps : [],
            base_time: data.base_time || null,
            step_hours: data.step_hours || null,
          });
        }
      } catch (err) {
        // 静默失败，沿用数字时间步
        console.warn('Failed to load time metadata', err);
      }
    };
    loadTimeMeta();
  }, []);

  // 将时间步转换为可读时间标签
  const formatStepLabel = useCallback((step) => {
    // 如果有基准时间与步长，直接计算
    if (timeMeta.base_time && timeMeta.step_hours) {
      const base = new Date(timeMeta.base_time);
      if (!isNaN(base.getTime())) {
        const ms = base.getTime() + Number(step || 0) * Number(timeMeta.step_hours) * 3600 * 1000;
        const d = new Date(ms);
        if (!isNaN(d.getTime())) {
          const pad = (n) => String(n).padStart(2, '0');
          return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:00 UTC`;
        }
      }
    }
    // 若timesteps数组存在，则展示原始值
    if (Array.isArray(timeMeta.timesteps) && timeMeta.timesteps.length > step) {
      return `时间 ${timeMeta.timesteps[step]}`;
    }
    // 兜底
    return `时间步 ${step}`;
  }, [timeMeta]);

  // 加载单个时间步的台风位置
  const loadSingleTimeStep = useCallback(async (timeStep) => {
    setLoading(true);
    setError(null);
    setTrackingProgress({ message: `正在加载时间步 ${timeStep}...`, progress: 0 });
    setTracks({});

    try {
      const response = await fetch(`${API_BASE_URL}/api/typhoon/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          time_step: timeStep,
          data_quality: -9,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      if (data.success && data.centers) {
        // 将单个时间步的台风中心转换为tracks格式
        const singleTracks = {};
        data.centers.forEach((center, idx) => {
          const typhoonId = `台风_${idx + 1}`;
          singleTracks[typhoonId] = [center];
        });
        setTracks(singleTracks);
        setCurrentTimeStep(timeStep);
        setLoading(false);
        setTrackingProgress(null);
        
        // 触发地图自动定位（通过更新positions触发）
        // 这会在getCurrentPositions计算完成后自动触发MapAutoCenter
      } else {
        throw new Error(data.error || '加载失败');
      }
    } catch (err) {
      setError(err.message || '加载失败');
      setLoading(false);
      setTrackingProgress(null);
    }
  }, []);

  // 开始追踪（全部时间步）
  const startTracking = useCallback(async () => {
    setLoading(true);
    setError(null);
    setTrackingProgress({ message: '正在初始化追踪...', progress: 0 });
    setTracks({});

    try {
      const response = await fetch(`${API_BASE_URL}/api/typhoon/track`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          start_time_step: 4320,
          end_time_step: 6470,
          data_quality: -9,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      // 使用流式读取来获取进度更新
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() && line.startsWith('data: ')) {
            try {
              const jsonStr = line.substring(6); // 移除 "data: " 前缀
              const data = JSON.parse(jsonStr);
              
              if (data.progress) {
                setTrackingProgress(data.progress);
              }
              if (data.tracks) {
                setTracks(data.tracks);
                if (data.success) {
                  setLoading(false);
                  setTrackingProgress(null);
                  // 追踪完成后，自动定位到初始时间步的台风中心
                  setCurrentTimeStep(4320);
                }
              }
              if (data.error) {
                setError(data.error);
                setLoading(false);
                setTrackingProgress(null);
                break;
              }
            } catch (e) {
              // 忽略JSON解析错误
              console.warn('Failed to parse SSE data:', line, e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message || '追踪失败');
      setLoading(false);
      setTrackingProgress(null);
    }
  }, []);

  // 获取当前时间步的台风位置
  const getCurrentPositions = useCallback(() => {
    const positions = {};
    // 根据实际数据集，降分辨率后的网格大小约为 810x1080
    // 默认经纬度范围：lat 10-40, lon 100-130
    const latStart = 10;
    const latEnd = 40;
    const lonStart = 100;
    const lonEnd = 130;
    const gridRows = 810;
    const gridCols = 1080;
    
    Object.keys(tracks).forEach((typhoonId) => {
      const trajectory = tracks[typhoonId];
      if (!trajectory || trajectory.length === 0) return;
      
      // 在单个时间步模式下，数据只存储在索引0
      // 在全部追踪模式下，数据按时间步索引存储
      let positionData = null;
      
      if (trackingMode === 'single') {
        // 单个时间步模式：只使用索引0的数据
        if (trajectory[0] && trajectory[0] !== null) {
          positionData = trajectory[0];
        }
      } else {
        // 全部追踪模式：使用currentTimeStep索引的数据
        if (trajectory[currentTimeStep] && trajectory[currentTimeStep] !== null) {
          positionData = trajectory[currentTimeStep];
        }
      }
      
      if (positionData) {
        const [row, col, timeStep] = positionData;
        // 将网格坐标转换为经纬度
        const lat = latStart + (row / gridRows) * (latEnd - latStart);
        const lng = lonStart + (col / gridCols) * (lonEnd - lonStart);
        positions[typhoonId] = { lat, lng, row, col, timeStep };
      }
    });
    return positions;
  }, [tracks, currentTimeStep, trackingMode]);

  // 获取台风轨迹线
  const getTrajectoryLines = useCallback(() => {
    const lines = {};
    // 根据实际数据集，降分辨率后的网格大小约为 810x1080
    const latStart = 10;
    const latEnd = 40;
    const lonStart = 100;
    const lonEnd = 130;
    const gridRows = 810;
    const gridCols = 1080;
    
    Object.keys(tracks).forEach((typhoonId) => {
      const trajectory = tracks[typhoonId];
      const points = [];
      trajectory.forEach((pos, idx) => {
        if (pos && pos !== null && idx <= currentTimeStep) {
          const [row, col] = pos;
          const lat = latStart + (row / gridRows) * (latEnd - latStart);
          const lng = lonStart + (col / gridCols) * (lonEnd - lonStart);
          points.push([lat, lng]);
        }
      });
      if (points.length > 1) {
        lines[typhoonId] = points;
      }
    });
    return lines;
  }, [tracks, currentTimeStep]);

  // 时间轴组件（滑块形式）
  const TimeAxis = () => {
    const [hoveredStep, setHoveredStep] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const sliderRef = useRef(null);
    
    // 处理时间步变化
    const handleTimeStepChange = (step) => {
      const clampedStep = Math.max(4320, Math.min(6470, parseInt(step, 10)));
      setCurrentTimeStep(clampedStep);
      if (trackingMode === 'single') {
        // 单个时间步模式下，拖动时自动加载对应时间步的数据
        loadSingleTimeStep(clampedStep);
      }
      // 全部追踪模式下，只需要更新currentTimeStep，getCurrentPositions会自动获取对应时间步的数据
    };
    
    // 处理滑块拖动
    const handleSliderChange = (e) => {
      const step = parseInt(e.target.value, 10);
      handleTimeStepChange(step);
    };
    
    // 处理滑块鼠标移动（显示悬停提示）
    const handleSliderMouseMove = (e) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const step = Math.round(percentage * (6470 - 4320) + 4320);
      setHoveredStep(Math.max(4320, Math.min(6470, step)));
    };
    
    // 计算滑块上标记点的位置
    const getMarkerPosition = (step) => {
      return ((step - 4320) / (6470 - 4320)) * 100;
    };

    return (
      <div
        style={{
          position: 'absolute',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          borderRadius: '12px',
          padding: '20px 32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
          zIndex: 1000,
          width: '90%',
          maxWidth: '900px',
        }}
      >
        {/* 时间步标签和当前值 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <span style={{ fontSize: '14px', fontWeight: 600, color: '#2d3748' }}>
            时间步
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '16px', fontWeight: 700, color: '#4299e1' }}>
              {formatStepLabel(currentTimeStep)}
            </span>
            <span style={{ fontSize: '14px', color: '#64748b' }}>
              / 6470
            </span>
          </div>
        </div>
        
        {/* 滑块容器 */}
        <div
          ref={sliderRef}
          style={{ position: 'relative', width: '100%', marginBottom: '12px' }}
          onMouseMove={handleSliderMouseMove}
          onMouseLeave={() => setHoveredStep(null)}
        >
          {/* 悬停提示 */}
          {hoveredStep !== null && (
            <div
              style={{
                position: 'absolute',
                left: `${getMarkerPosition(hoveredStep)}%`,
                bottom: '30px',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                color: 'white',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 600,
                pointerEvents: 'none',
                whiteSpace: 'nowrap',
                zIndex: 1001,
              }}
            >
            {formatStepLabel(hoveredStep)}
            </div>
          )}
          
          {/* 滑块 */}
          <input
            type="range"
            min="4320"
            max="6470"
            value={currentTimeStep}
            onChange={handleSliderChange}
            onMouseDown={() => setIsDragging(true)}
            onMouseUp={() => setIsDragging(false)}
            style={{
              width: '100%',
              height: '8px',
              borderRadius: '4px',
              background: 'linear-gradient(to right, #4299e1 0%, #4299e1 ' + getMarkerPosition(currentTimeStep) + '%, #e2e8f0 ' + getMarkerPosition(currentTimeStep) + '%, #e2e8f0 100%)',
              outline: 'none',
              cursor: 'pointer',
              WebkitAppearance: 'none',
              appearance: 'none',
            }}
          />
          
          {/* 自定义滑块样式 */}
          <style>{`
            input[type="range"]::-webkit-slider-thumb {
              -webkit-appearance: none;
              appearance: none;
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: #4299e1;
              cursor: pointer;
              border: 3px solid white;
              box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
              transition: all 0.2s ease;
            }
            
            input[type="range"]::-webkit-slider-thumb:hover {
              background: #3182ce;
              transform: scale(1.1);
            }
            
            input[type="range"]::-moz-range-thumb {
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: #4299e1;
              cursor: pointer;
              border: 3px solid white;
              box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
              transition: all 0.2s ease;
            }
            
            input[type="range"]::-moz-range-thumb:hover {
              background: #3182ce;
              transform: scale(1.1);
            }
          `}</style>
          
          {/* 时间步标记点（每500个时间步一个标记） */}
          <div style={{ position: 'absolute', top: '10px', left: 0, right: 0, display: 'flex', justifyContent: 'space-between', pointerEvents: 'none' }}>
            {[4320, 4820, 5320, 5820, 6320, 6470].map((step) => (
              <div
                key={step}
                style={{
                  position: 'absolute',
                  left: `${getMarkerPosition(step)}%`,
                  transform: 'translateX(-50%)',
                  fontSize: '10px',
                  color: '#64748b',
                  fontWeight: 500,
                }}
              >
                {formatStepLabel(step)}
              </div>
            ))}
          </div>
        </div>
        
        {/* 控制按钮 */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
          <button
            onClick={() => {
              const prevStep = Math.max(4320, currentTimeStep - 1);
              handleTimeStepChange(prevStep);
            }}
            disabled={currentTimeStep === 4320}
            style={{
              backgroundColor: currentTimeStep === 4320 ? '#cbd5e0' : '#4299e1',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '6px 12px',
              fontSize: '14px',
              cursor: currentTimeStep === 4320 ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            ← 上一步
          </button>
          <button
            onClick={() => {
              const nextStep = Math.min(6470, currentTimeStep + 1);
              handleTimeStepChange(nextStep);
            }}
            disabled={currentTimeStep === 6470}
            style={{
              backgroundColor: currentTimeStep === 6470 ? '#cbd5e0' : '#4299e1',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '6px 12px',
              fontSize: '14px',
              cursor: currentTimeStep === 6470 ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            下一步 →
          </button>
          {trackingMode === 'all' && (
            <button
              onClick={() => {
                let step = 4320;
                const interval = setInterval(() => {
                  setCurrentTimeStep(step);
                  if (step < 6470) {
                    step++;
                  } else {
                    clearInterval(interval);
                  }
                }, 200); // 每200ms切换一次
              }}
              style={{
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                padding: '6px 12px',
                fontSize: '14px',
                cursor: 'pointer',
                marginLeft: '8px',
                transition: 'all 0.2s ease',
              }}
            >
              ▶ 播放动画
            </button>
          )}
        </div>
      </div>
    );
  };

  const currentPositions = getCurrentPositions();
  const trajectoryLines = getTrajectoryLines();

  // 缩小模式样式
  if (isMinimized) {
    return (
      <div
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '20px',
          width: '400px',
          height: '300px',
          backgroundColor: '#ffffff',
          zIndex: 3500,
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '2px solid #4299e1',
        }}
      >
        {/* 缩小窗口标题栏 */}
        <div
          style={{
            backgroundColor: '#4299e1',
            color: 'white',
            padding: '8px 12px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'move',
          }}
        >
          <span style={{ fontSize: '14px', fontWeight: 600 }}>台风追踪（已缩小）</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={onRestore}
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '4px 8px',
                fontSize: '12px',
                cursor: 'pointer',
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.3)'}
              onMouseOut={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.2)'}
            >
              恢复
            </button>
            <button
              onClick={onBack}
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '4px 8px',
                fontSize: '12px',
                cursor: 'pointer',
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.3)'}
              onMouseOut={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.2)'}
            >
              关闭
            </button>
          </div>
        </div>
        {/* 缩小后的地图预览 */}
        <div style={{ flex: 1, position: 'relative' }}>
          <MapContainer
            style={{ width: '100%', height: '100%' }}
            center={[25, 120]}
            zoom={4}
            maxZoom={19}
            minZoom={3}
            scrollWheelZoom={false}
            dragging={true}
          >
            <TileLayer
              attribution='&copy; <a href="https://opentopomap.org">OpenTopoMap</a> contributors'
              url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
              tileSize={512}
              zoomOffset={-1}
            />
            <MapAutoCenter positions={currentPositions} />
            {/* 绘制轨迹线 */}
            {Object.keys(trajectoryLines).map((typhoonId) => (
              <Polyline
                key={`trajectory-${typhoonId}`}
                positions={trajectoryLines[typhoonId]}
                pathOptions={{
                  color: selectedTyphoon === typhoonId ? '#dc2626' : '#94a3b8',
                  weight: 1,
                  opacity: 0.5,
                }}
              />
            ))}
            {/* 绘制当前时间步的台风位置 */}
            {Object.keys(currentPositions).map((typhoonId) => {
              const pos = currentPositions[typhoonId];
              const isActive = selectedTyphoon === typhoonId || selectedTyphoon === null;
              return (
                <React.Fragment key={typhoonId}>
                  <Marker
                    position={[pos.lat, pos.lng]}
                    icon={createTyphoonIcon(typhoonId.replace('台风_', ''), isActive)}
                    eventHandlers={{
                      click: () => {
                        setSelectedTyphoon(selectedTyphoon === typhoonId ? null : typhoonId);
                      },
                    }}
                  />
                </React.Fragment>
              );
            })}
          </MapContainer>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: '#ffffff',
        zIndex: 2000,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* 顶部控制栏 */}
      <div
        style={{
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          padding: '16px 24px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 1001,
        }}
      >
        <h1 style={{ margin: 0, fontSize: '1.5rem', color: '#2d3748', fontWeight: 600 }}>
          台风追踪界面
        </h1>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <label style={{ fontSize: '14px', color: '#2d3748', fontWeight: 500 }}>
              模式:
            </label>
            <select
              value={trackingMode}
              onChange={(e) => {
                setTrackingMode(e.target.value);
                setTracks({});
                setCurrentTimeStep(4320);
              }}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e0',
                fontSize: '14px',
                cursor: 'pointer',
              }}
            >
              <option value="all">全部追踪 (4320-6470)</option>
              <option value="single">单个时间步</option>
            </select>
          </div>
          
          {trackingMode === 'single' ? (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <label style={{ fontSize: '14px', color: '#2d3748', fontWeight: 500 }}>
                时间步:
              </label>
              <input
                type="number"
                min="4320"
                max="6470"
                value={currentTimeStep}
                onChange={(e) => {
                  const step = parseInt(e.target.value, 10) || 4320;
                  const clampedStep = Math.max(4320, Math.min(6470, step));
                  setCurrentTimeStep(clampedStep);
                }}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    loadSingleTimeStep(currentTimeStep);
                  }
                }}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  border: '1px solid #cbd5e0',
                  fontSize: '14px',
                  width: '80px',
                }}
              />
              <button
                onClick={() => loadSingleTimeStep(currentTimeStep)}
                disabled={loading}
                style={{
                  backgroundColor: loading ? '#94a3b8' : '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '10px 20px',
                  fontSize: '14px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                }}
              >
                {loading ? '加载中...' : '加载时间步'}
              </button>
            </div>
          ) : (
            <button
              onClick={startTracking}
              disabled={loading}
              style={{
                backgroundColor: loading ? '#94a3b8' : '#4299e1',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '10px 20px',
                fontSize: '14px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: 600,
              }}
            >
              {loading ? '追踪中...' : '开始追踪 (4320-6470)'}
            </button>
          )}
          <button
            onClick={onBack}
            style={{
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              padding: '10px 20px',
              fontSize: '14px',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            返回地图
          </button>
        </div>
      </div>

      {/* 进度提示 */}
      {trackingProgress && (
        <div
          style={{
            backgroundColor: 'rgba(66, 153, 225, 0.1)',
            padding: '12px 24px',
            borderBottom: '1px solid #e2e8f0',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '20px',
                height: '20px',
                border: '3px solid #4299e1',
                borderTop: '3px solid transparent',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }}
            />
            <span style={{ fontSize: '14px', color: '#2d3748' }}>
              {trackingProgress.message || '处理中...'}
            </span>
            {trackingProgress.progress !== undefined && (
              <span style={{ fontSize: '14px', color: '#64748b', marginLeft: 'auto' }}>
                {Math.round(trackingProgress.progress)}%
              </span>
            )}
          </div>
        </div>
      )}

      {/* 错误提示 */}
      {error && (
        <div
          style={{
            backgroundColor: '#fee2e2',
            padding: '12px 24px',
            borderBottom: '1px solid #fecaca',
            color: '#dc2626',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      {/* 地图容器 */}
      <div style={{ flex: 1, position: 'relative' }}>
        <MapContainer
          style={{ width: '100%', height: '100%' }}
          center={[25, 120]}
          zoom={5}
          maxZoom={19}
          minZoom={3}
          scrollWheelZoom={true}
          dragging={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://opentopomap.org">OpenTopoMap</a> contributors &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
            tileSize={512}
            zoomOffset={-1}
          />
          
          {/* 地图自动定位组件 */}
          <MapAutoCenter positions={currentPositions} />

          {/* 绘制轨迹线 */}
          {Object.keys(trajectoryLines).map((typhoonId) => (
            <Polyline
              key={`trajectory-${typhoonId}`}
              positions={trajectoryLines[typhoonId]}
              pathOptions={{
                color: selectedTyphoon === typhoonId ? '#dc2626' : '#94a3b8',
                weight: 2,
                opacity: 0.6,
              }}
            />
          ))}

          {/* 绘制当前时间步的台风位置 */}
          {Object.keys(currentPositions).map((typhoonId) => {
            const pos = currentPositions[typhoonId];
            const isActive = selectedTyphoon === typhoonId || selectedTyphoon === null;
            return (
              <React.Fragment key={typhoonId}>
                <Marker
                  position={[pos.lat, pos.lng]}
                  icon={createTyphoonIcon(typhoonId.replace('台风_', ''), isActive)}
                  eventHandlers={{
                    click: () => {
                      setSelectedTyphoon(selectedTyphoon === typhoonId ? null : typhoonId);
                    },
                    dblclick: () => {
                      // 双击打开台风详情页
                      if (onTyphoonClick) {
                        // 从台风ID中提取数字（如"台风_1" -> 1）
                        const typhoonNum = parseInt(typhoonId.replace('台风_', ''), 10);
                        if (!isNaN(typhoonNum)) {
                          onTyphoonClick(typhoonNum, false);
                        }
                      }
                    },
                  }}
                />
                <Circle
                  center={[pos.lat, pos.lng]}
                  radius={300000} // 300公里半径（增大）
                  pathOptions={{
                    color: isActive ? '#dc2626' : '#94a3b8',
                    fillColor: isActive ? '#dc2626' : '#94a3b8',
                    fillOpacity: 0.15,
                    weight: 3,
                    opacity: 0.5,
                  }}
                />
              </React.Fragment>
            );
          })}
        </MapContainer>

        {/* 时间轴 */}
        <TimeAxis />

        {/* 台风信息面板 */}
        {selectedTyphoon && currentPositions[selectedTyphoon] && (
          <div
            style={{
              position: 'absolute',
              top: '80px',
              right: '24px',
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              borderRadius: '12px',
              padding: '16px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
              zIndex: 1000,
              minWidth: '200px',
            }}
          >
            <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', color: '#2d3748' }}>
              {selectedTyphoon}
            </h3>
            <div style={{ fontSize: '14px', color: '#64748b' }}>
              <p style={{ margin: '4px 0' }}>
                <strong>时间步:</strong> {formatStepLabel(currentTimeStep)}
              </p>
              <p style={{ margin: '4px 0' }}>
                <strong>位置:</strong> {currentPositions[selectedTyphoon].lat.toFixed(2)}°N,{' '}
                {currentPositions[selectedTyphoon].lng.toFixed(2)}°E
              </p>
              <p style={{ margin: '4px 0' }}>
                <strong>网格:</strong> ({currentPositions[selectedTyphoon].row},{' '}
                {currentPositions[selectedTyphoon].col})
              </p>
            </div>
            {onTyphoonClick && (
              <button
                onClick={() => {
                  const typhoonNum = parseInt(selectedTyphoon.replace('台风_', ''), 10);
                  if (!isNaN(typhoonNum)) {
                    onTyphoonClick(typhoonNum, false);
                  }
                }}
                style={{
                  marginTop: '12px',
                  width: '100%',
                  backgroundColor: '#4299e1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  fontSize: '14px',
                  cursor: 'pointer',
                  fontWeight: 600,
                  transition: 'all 0.2s ease',
                }}
                onMouseOver={(e) => e.target.style.backgroundColor = '#3182ce'}
                onMouseOut={(e) => e.target.style.backgroundColor = '#4299e1'}
              >
                查看详情
              </button>
            )}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default TyphoonTracking;

