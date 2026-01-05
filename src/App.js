import React, { useEffect, useState, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Circle, Rectangle, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { TyphoonPage, VisualizationHistoryPanel } from './wind';
import TyphoonTracking from './TyphoonTracking';
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// 数据提取器面板组件
const DataExtractorPanel = ({
  visible,
  onClose,
  API_BASE_URL,
  onExtractSuccess
}) => {
  const [formData, setFormData] = useState({
    lon_min: '',
    lon_max: '',
    lat_min: '',
    lat_max: '',
    time_step: '0',
    layer_min: '',
    layer_max: '',
    variables: '',
    save_data: false,
    save_path: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [availableVariables, setAvailableVariables] = useState({ atmosphere: [], ocean: [] });
  const [showHelp, setShowHelp] = useState(false);

  // 加载可用变量列表
  useEffect(() => {
    if (visible) {
      fetch(`${API_BASE_URL}/api/data/extract/variables`)
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setAvailableVariables({
              atmosphere: data.atmosphere_variables || [],
              ocean: data.ocean_variables || []
            });
          }
        })
        .catch(err => console.error('Failed to load variables:', err));
    }
  }, [visible, API_BASE_URL]);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError(null);
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // 验证必需参数
      if (!formData.lon_min || !formData.lon_max || !formData.lat_min || !formData.lat_max || !formData.time_step) {
        throw new Error('请填写所有必需参数（经纬范围和时间步）');
      }

      // 准备请求数据
      const requestData = {
        lon_min: parseFloat(formData.lon_min),
        lon_max: parseFloat(formData.lon_max),
        lat_min: parseFloat(formData.lat_min),
        lat_max: parseFloat(formData.lat_max),
        time_step: parseInt(formData.time_step, 10),
        layer_min: formData.layer_min ? parseInt(formData.layer_min, 10) : null,
        layer_max: formData.layer_max ? parseInt(formData.layer_max, 10) : null,
        variables: formData.variables ? formData.variables.split(',').map(v => v.trim()).filter(v => v) : null,
        save_data: formData.save_data,
        save_path: formData.save_path || null
      };

      const response = await fetch(`${API_BASE_URL}/api/data/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || '数据提取失败');
      }

      setResult(data);
      
      // 提取成功后，通知父组件在地图上标记区域
      if (onExtractSuccess) {
        onExtractSuccess({
          lon_min: requestData.lon_min,
          lon_max: requestData.lon_max,
          lat_min: requestData.lat_min,
          lat_max: requestData.lat_max,
          time_step: requestData.time_step,
          summary: data.summary
        });
      }
    } catch (err) {
      setError(err.message || '提取数据时发生错误');
      console.error('Data extraction error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!visible) return null;

  const fieldStyle = {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    fontSize: '12px',
    color: '#4a5568',
  };

  return (
    <div
      style={{
        position: 'absolute',
        top: '80px',
        left: '24px',
        zIndex: 1100,
        background: 'rgba(255,255,255,0.97)',
        padding: 20,
        borderRadius: 12,
        boxShadow: '0 8px 24px rgba(15,23,42,0.2)',
        width: 420,
        maxHeight: '85vh',
        overflowY: 'auto',
        border: '1px solid #e2e8f0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ margin: 0, color: '#1f2937', fontSize: '18px' }}>🌊 大气海洋数据提取</h3>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => setShowHelp(!showHelp)}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: 16,
              cursor: 'pointer',
              color: '#4299e1',
              padding: '4px 8px'
            }}
            title="显示帮助"
          >
            ❓
          </button>
          <button
            onClick={onClose}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: 20,
              cursor: 'pointer',
              color: '#4b5563'
            }}
          >
            ×
          </button>
        </div>
      </div>

      {showHelp && (
        <div style={{
          marginBottom: 16,
          padding: 12,
          background: '#f0f9ff',
          borderRadius: 8,
          border: '1px solid #bae6fd',
          fontSize: '12px',
          color: '#0c4a6e'
        }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>📖 参数说明：</h4>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            <li><strong>经纬范围</strong>：左下角(经度,纬度) 和 右上角(经度,纬度)</li>
            <li><strong>时间步</strong>：数据的时间索引（从0开始）</li>
            <li><strong>层数范围</strong>：可选，留空则提取全部层。大气层0-50，海洋层0-89</li>
            <li><strong>变量</strong>：可选，用逗号分隔。留空则提取全部变量</li>
            <li><strong>保存数据</strong>：是否保存到本地文件</li>
          </ul>
          <div style={{ marginTop: 8 }}>
            <strong>可用变量：</strong>
            <div style={{ marginTop: 4 }}>
              <span style={{ color: '#0369a1' }}>大气：</span> {availableVariables.atmosphere.join(', ')}
            </div>
            <div style={{ marginTop: 4 }}>
              <span style={{ color: '#0369a1' }}>海洋：</span> {availableVariables.ocean.join(', ')}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div style={{
          marginBottom: 16,
          padding: 12,
          background: '#fef2f2',
          borderRadius: 8,
          border: '1px solid #fecaca',
          color: '#dc2626',
          fontSize: '13px'
        }}>
          ❌ {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* 经纬范围 */}
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#374151', fontWeight: 600 }}>
              【1】经纬范围 <span style={{ color: '#dc2626' }}>*</span>
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <label style={fieldStyle}>
                左下角经度 (lon_min)
                <input
                  type="number"
                  required
                  value={formData.lon_min}
                  onChange={(e) => handleInputChange('lon_min', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  step="0.1"
                  placeholder="-180 ~ 180"
                />
              </label>
              <label style={fieldStyle}>
                左下角纬度 (lat_min)
                <input
                  type="number"
                  required
                  value={formData.lat_min}
                  onChange={(e) => handleInputChange('lat_min', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  step="0.1"
                  placeholder="-90 ~ 90"
                />
              </label>
              <label style={fieldStyle}>
                右上角经度 (lon_max)
                <input
                  type="number"
                  required
                  value={formData.lon_max}
                  onChange={(e) => handleInputChange('lon_max', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  step="0.1"
                  placeholder="-180 ~ 180"
                />
              </label>
              <label style={fieldStyle}>
                右上角纬度 (lat_max)
                <input
                  type="number"
                  required
                  value={formData.lat_max}
                  onChange={(e) => handleInputChange('lat_max', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  step="0.1"
                  placeholder="-90 ~ 90"
                />
              </label>
            </div>
          </div>

          {/* 时间步 */}
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#374151', fontWeight: 600 }}>
              【2】时间步 <span style={{ color: '#dc2626' }}>*</span>
            </h4>
            <label style={fieldStyle}>
              时间步索引 (time_step)
              <input
                type="number"
                required
                value={formData.time_step}
                onChange={(e) => handleInputChange('time_step', e.target.value)}
                style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                min="0"
                placeholder="0"
              />
            </label>
          </div>

          {/* 层数范围 */}
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#374151', fontWeight: 600 }}>
              【3】层数范围（可选）
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <label style={fieldStyle}>
                层数下界 (layer_min)
                <input
                  type="number"
                  value={formData.layer_min}
                  onChange={(e) => handleInputChange('layer_min', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  placeholder="留空=全部"
                />
              </label>
              <label style={fieldStyle}>
                层数上界 (layer_max)
                <input
                  type="number"
                  value={formData.layer_max}
                  onChange={(e) => handleInputChange('layer_max', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  placeholder="留空=全部"
                />
              </label>
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: '#64748b' }}>
              提示：大气层0-50，海洋层0-89。留空则提取全部层
            </p>
          </div>

          {/* 变量 */}
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#374151', fontWeight: 600 }}>
              【4】变量列表（可选）
            </h4>
            <label style={fieldStyle}>
              变量（用逗号分隔，留空=全部）
              <input
                type="text"
                value={formData.variables}
                onChange={(e) => handleInputChange('variables', e.target.value)}
                style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                placeholder="例如: U,V,T 或留空提取全部"
              />
            </label>
          </div>

          {/* 保存选项 */}
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#374151', fontWeight: 600 }}>
              【5】保存选项（可选）
            </h4>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={formData.save_data}
                onChange={(e) => handleInputChange('save_data', e.target.checked)}
                style={{ width: 18, height: 18, cursor: 'pointer' }}
              />
              <span style={{ fontSize: '13px', color: '#4a5568' }}>保存数据到本地</span>
            </label>
            {formData.save_data && (
              <label style={{ ...fieldStyle, marginTop: 8 }}>
                保存路径（留空=自动生成）
                <input
                  type="text"
                  value={formData.save_path}
                  onChange={(e) => handleInputChange('save_path', e.target.value)}
                  style={{ padding: '6px 8px', borderRadius: 6, border: '1px solid #cbd5e0' }}
                  placeholder="例如: data.npz 或 data.nc"
                />
              </label>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '8px 14px',
              borderRadius: 8,
              border: '1px solid #e2e8f0',
              background: 'white',
              color: '#4b5563',
              cursor: 'pointer'
            }}
          >
            取消
          </button>
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: '8px 16px',
              borderRadius: 8,
              border: 'none',
              background: loading ? '#94a3b8' : '#10b981',
              color: 'white',
              cursor: loading ? 'not-allowed' : 'pointer',
              minWidth: 90,
            }}
          >
            {loading ? '提取中...' : '开始提取'}
          </button>
        </div>
      </form>

      {result && (
        <div style={{
          marginTop: 20,
          padding: 16,
          background: '#f0fdf4',
          borderRadius: 8,
          border: '1px solid #86efac'
        }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#166534' }}>✅ 提取成功！</h4>
          {result.summary && (
            <div style={{ fontSize: '12px', color: '#166534' }}>
              {result.summary.coordinates && (
                <div style={{ marginBottom: 8 }}>
                  <strong>坐标点数：</strong>{result.summary.coordinates.count}
                  <br />
                  <strong>经度范围：</strong>{result.summary.coordinates.lon_range[0].toFixed(2)} ~ {result.summary.coordinates.lon_range[1].toFixed(2)}
                  <br />
                  <strong>纬度范围：</strong>{result.summary.coordinates.lat_range[0].toFixed(2)} ~ {result.summary.coordinates.lat_range[1].toFixed(2)}
                </div>
              )}
              {result.summary.layers && (
                <div style={{ marginBottom: 8 }}>
                  <strong>层数：</strong>{result.summary.layers.count} 层 ({result.summary.layers.range[0]} ~ {result.summary.layers.range[1]})
                </div>
              )}
              <div>
                <strong>变量数据：</strong>
                {Object.keys(result.summary).filter(k => k !== 'coordinates' && k !== 'layers').map(varName => (
                  <div key={varName} style={{ marginTop: 4, paddingLeft: 12 }}>
                    <strong>{varName}:</strong> shape={JSON.stringify(result.summary[varName].shape)}, 
                    min={result.summary[varName].min.toFixed(4)}, 
                    max={result.summary[varName].max.toFixed(4)}, 
                    mean={result.summary[varName].mean.toFixed(4)}
                  </div>
                ))}
              </div>
              {result.save_path && (
                <div style={{ marginTop: 8, padding: 8, background: '#dcfce7', borderRadius: 4 }}>
                  <strong>已保存到：</strong>{result.save_path} ({result.save_format})
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// 修复 Leaflet 图标路径问题
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// 时间轴组件（用于主地图）
const TimeAxis = ({ currentTime, onTimeChange }) => {
  return (
    <div style={{
      position: 'absolute',
      top: '80px',
      right: '24px',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderRadius: '8px',
      padding: '12px 16px',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.15)',
      zIndex: 1000,
      display: 'flex',
      gap: '12px',
      alignItems: 'center'
    }}>
      <span style={{ fontSize: '14px', fontWeight: 600, color: '#2d3748', marginRight: '8px' }}>时间轴：</span>
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
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}
        onMouseOver={(e) => {
          if (currentTime !== 1) {
            e.target.style.backgroundColor = '#cbd5e0';
          }
        }}
        onMouseOut={(e) => {
          if (currentTime !== 1) {
            e.target.style.backgroundColor = '#e2e8f0';
          }
        }}
      >
        时间1
      </button>
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

// 占位：你要跳转的「其他界面」（替换为你的实际界面组件）
const TargetPage = ({ selectedData, onBack }) => {
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: '#ffffff',
      zIndex: 2000,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      boxSizing: 'border-box'
    }}>
      <h1 style={{ color: '#2d3748', marginBottom: '30px' }}>选择完成 → 跳转目标界面</h1>
      
      {/* 展示选择的数据（可根据需求传递给目标界面） */}
      <div style={{
        backgroundColor: '#f8f9fa',
        padding: '20px',
        borderRadius: '8px',
        width: '80%',
        maxWidth: '600px',
        marginBottom: '30px'
      }}>
        <h3 style={{ color: '#4a5568', margin: '0 0 16px 0' }}>选择区域信息</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <p style={{ margin: '4px 0', fontSize: '13px', color: '#718096' }}>区域尺寸</p>
            <p style={{ margin: '0', fontSize: '14px', color: '#2d3748' }}>
              {Math.round(selectedData.screenRect.width)} × {Math.round(selectedData.screenRect.height)} px
            </p>
          </div>
          <div>
            <p style={{ margin: '4px 0', fontSize: '13px', color: '#718096' }}>中心经纬度</p>
            <p style={{ margin: '0', fontSize: '14px', color: '#2d3748' }}>
              {selectedData.mapRect.center.lat.toFixed(6)}, {selectedData.mapRect.center.lng.toFixed(6)}
            </p>
          </div>
        </div>
      </div>

      {/* 返回地图按钮 */}
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
        返回地图重新选择
      </button>
    </div>
  );
};

// 全局样式
const globalStyle = `
  html, body, #root, .App {
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 0;
  }

  * {
    box-sizing: border-box;
    font-family: 'Arial', sans-serif;
  }

  /* 地图容器 */
  .leaflet-container {
    width: 100% !important;
    height: 100% !important;
    cursor: default;
    z-index: 1 !important;
  }

  .leaflet-control-zoom {
    z-index: 80 !important;
  }

  /* 3D小地图样式 */
  .mini-map-3d-container {
    position: absolute;
    bottom: 20px;
    right: 20px;
    width: 200px;
    height: 200px;
    border: 2px solid white;
    border-radius: 4px;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.4);
    z-index: 90;
    overflow: hidden;
  }

  .mini-map-3d-wrapper {
    width: 100% !important;
    height: 100% !important;
  }

  .mini-map-3d-close-btn {
    position: absolute;
    top: 5px;
    right: 5px;
    width: 22px;
    height: 22px;
    background-color: rgba(0, 0, 0, 0.7);
    border: none;
    border-radius: 50%;
    color: white;
    font-size: 14px;
    line-height: 22px;
    text-align: center;
    cursor: pointer;
    z-index: 100;
    padding: 0;
  }

  .mini-map-3d-close-btn:hover {
    background-color: #dc2626;
  }

  .mini-map-3d-title {
    position: absolute;
    top: 5px;
    left: 10px;
    color: white;
    font-size: 12px;
    font-weight: 600;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
    z-index: 100;
  }

  /* 选择按钮样式（醒目可见） */
  .select-btn {
    position: absolute;
    top: 80px;
    left: 24px;
    background-color: #4299e1;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    z-index: 999 !important;
    box-shadow: 0 4px 12px rgba(66, 153, 225, 0.5);
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    border: 2px solid white;
  }

  .select-btn:hover {
    background-color: #3182ce;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(66, 153, 225, 0.6);
  }

  .select-btn:active {
    transform: translateY(0);
  }

  /* 选择模式遮罩层 */
  .select-mask {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.6);
    z-index: 999;
    pointer-events: none;
    display: none;
  }

  /* QQ截图式矩形选择框 */
  .select-selection {
    position: absolute;
    border: 2px dashed #ffffff;
    background-color: rgba(255, 255, 255, 0.1);
    z-index: 1000;
    pointer-events: none;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.6);
    display: none;
  }

  /* 选框边角调整点 */
  .selection-handle {
    position: absolute;
    width: 12px;
    height: 12px;
    background-color: #4299e1;
    border: 2px solid white;
    border-radius: 50%;
    z-index: 1001;
    pointer-events: auto;
    cursor: nwse-resize;
  }

  .selection-handle-tl { top: -6px; left: -6px; }
  .selection-handle-tr { top: -6px; right: -6px; }
  .selection-handle-bl { bottom: -6px; left: -6px; }
  .selection-handle-br { bottom: -6px; right: -6px; }

  /* 操作提示 */
  .select-tip {
    position: fixed;
    bottom: 50px;
    left: 50%;
    transform: translateX(-50%);
    background-color: rgba(0, 0, 0, 0.8);
    color: #ffffff;
    padding: 10px 20px;
    border-radius: 20px;
    font-size: 14px;
    z-index: 1002;
    pointer-events: none;
    display: none;
  }

  /* 退出选择按钮 */
  .exit-select-btn {
    position: fixed;
    top: 80px;
    left: 24px;
    background-color: white;
    color: #4299e1;
    border: 2px solid #4299e1;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    cursor: pointer;
    z-index: 1003;
    display: none;
    transition: background-color 0.3s ease;
  }

  .exit-select-btn:hover {
    background-color: #e6f7ff;
  }

  /* 顶部标题栏 */
  .map-header {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background-color: rgba(255, 255, 255, 0.9);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    z-index: 90 !important;
    backdropFilter: blur(4px);
  }

  /* 加载动画 */
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  /* 历史记录面板滑入动画 */
  @keyframes slideUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

// 主地图定位标记
const LocationMarker = ({ currentPosition, accuracy }) => {
  const map = useMapEvents({});

  useEffect(() => {
    if (currentPosition) {
      map.setView(currentPosition, 13);
    }
  }, [currentPosition, map]);

  return (
    <>
      {currentPosition && (
        <>
          <Circle
            center={currentPosition}
            radius={accuracy}
            color="#4299e1"
            fillColor="#4299e1"
            fillOpacity={0.2}
          />
          <Marker position={currentPosition} />
        </>
      )}
    </>
  );
};

// 3D小地图定位标记
const MiniMap3dMarker = ({ currentPosition }) => {
  const markerIcon = L.divIcon({
    className: 'mini-map-3d-marker',
    html: `
      <div style="width: 12px; height: 12px; background: #ff0000; border-radius: 50%; 
                  border: 2px solid white; box-shadow: 0 0 8px #ff0000;"></div>
    `,
    iconSize: [12, 12],
    iconAnchor: [6, 6],
  });

  return currentPosition ? <Marker position={currentPosition} icon={markerIcon} /> : null;
};

// 台风按钮标记组件（通用）
const TyphoonButtonMarker = ({ position, label, onClick, onMouseEnter, onMouseLeave }) => {
  const markerRef = useRef(null);
  
  useEffect(() => {
    if (!markerRef.current) return;
    
    const marker = markerRef.current;
    const markerElement = marker.getElement();
    if (!markerElement) return;
    
    // 查找按钮元素
    const buttonElement = markerElement.querySelector('button');
    if (!buttonElement) return;
    
    // 添加鼠标悬停事件
    const handleMouseEnter = (e) => {
      buttonElement.style.backgroundColor = '#b91c1c';
      buttonElement.style.transform = 'scale(1.05)';
      if (onMouseEnter) {
        onMouseEnter();
      }
    };
    
    const handleMouseLeave = (e) => {
      buttonElement.style.backgroundColor = '#dc2626';
      buttonElement.style.transform = 'scale(1)';
      if (onMouseLeave) {
        onMouseLeave();
      }
    };
    
    buttonElement.addEventListener('mouseenter', handleMouseEnter);
    buttonElement.addEventListener('mouseleave', handleMouseLeave);
    
    return () => {
      buttonElement.removeEventListener('mouseenter', handleMouseEnter);
      buttonElement.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [onMouseEnter, onMouseLeave]);
  
  // 创建按钮元素
  const buttonElement = document.createElement('button');
  buttonElement.innerHTML = label;
  buttonElement.style.cssText = `
    background-color: #dc2626;
    color: white;
    border: 2px solid white;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.5);
    transition: all 0.3s ease;
    pointer-events: auto;
  `;
  
  // 添加点击事件
  buttonElement.addEventListener('click', (e) => {
    e.stopPropagation();
    if (onClick) {
      onClick();
    }
  });
  
  const buttonIcon = L.divIcon({
    className: 'typhoon-button-marker',
    html: buttonElement,
    iconSize: [80, 40],
    iconAnchor: [40, 20],
  });

  return (
    <Marker 
      ref={markerRef}
      position={position} 
      icon={buttonIcon}
      interactive={true}
    />
  );
};

// 3D地球小地图组件
const MiniMap3d = ({ currentPosition, isVisible, onToggleVisible }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const rotationRef = useRef({ x: -0.3, y: 0 });
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });

  // 将经纬度转换为3D坐标
  const latLngTo3D = (lat, lng, radius) => {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lng + 180) * (Math.PI / 180);
    
    return {
      x: -radius * Math.sin(phi) * Math.cos(theta),
      y: radius * Math.cos(phi),
      z: radius * Math.sin(phi) * Math.sin(theta)
    };
  };

  // 3D点旋转
  const rotatePoint = (point, rx, ry) => {
    // 绕Y轴旋转
    let x = point.x;
    let z = point.z;
    point.x = x * Math.cos(ry) + z * Math.sin(ry);
    point.z = -x * Math.sin(ry) + z * Math.cos(ry);
    
    // 绕X轴旋转
    let y = point.y;
    z = point.z;
    point.y = y * Math.cos(rx) - z * Math.sin(rx);
    point.z = y * Math.sin(rx) + z * Math.cos(rx);
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

  // 绘制3D地球
  const drawGlobe = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.45;
    const distance = 400;

    ctx.clearRect(0, 0, width, height);

    // 绘制背景
    ctx.fillStyle = '#1a202c';
    ctx.fillRect(0, 0, width, height);

    // 绘制地球球体基础（海洋）
    const oceanGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
    oceanGradient.addColorStop(0, 'rgba(30, 64, 175, 0.4)');
    oceanGradient.addColorStop(0.7, 'rgba(29, 78, 216, 0.3)');
    oceanGradient.addColorStop(1, 'rgba(30, 58, 138, 0.2)');
    
    ctx.fillStyle = oceanGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();

    // 绘制经纬线网格
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;

    // 绘制纬线
    for (let lat = -90; lat <= 90; lat += 30) {
      const points = [];
      for (let lng = -180; lng <= 180; lng += 5) {
        const point3D = latLngTo3D(lat, lng, radius);
        rotatePoint(point3D, rotationRef.current.x, rotationRef.current.y);
        point3D.z += distance;
        const proj = project(point3D, distance);
        if (proj.scale > 0) {
          points.push({ x: centerX + proj.x, y: centerY + proj.y });
        }
      }
      
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();
      }
    }

    // 绘制经线
    for (let lng = -180; lng <= 180; lng += 30) {
      const points = [];
      for (let lat = -90; lat <= 90; lat += 5) {
        const point3D = latLngTo3D(lat, lng, radius);
        rotatePoint(point3D, rotationRef.current.x, rotationRef.current.y);
        point3D.z += distance;
        const proj = project(point3D, distance);
        if (proj.scale > 0) {
          points.push({ x: centerX + proj.x, y: centerY + proj.y });
        }
      }
      
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();
      }
    }

    // 绘制地表纹理（大陆和地形）
    // 绘制大陆填充区域
    const continents = [
      // 亚洲（更详细）
      [[60, 30], [65, 40], [70, 50], [75, 60], [70, 70], [65, 80], [60, 90], [55, 100], [50, 110], [45, 120], [40, 130], [35, 140], [30, 150], [25, 160], [20, 150], [15, 140], [10, 130], [5, 120], [0, 110], [-5, 100], [0, 90], [5, 80], [10, 70], [15, 60], [20, 50], [25, 40], [30, 35], [35, 30], [40, 25], [45, 20], [50, 25], [55, 30]],
      // 欧洲
      [[70, -10], [75, 0], [80, 10], [75, 20], [70, 30], [65, 35], [60, 30], [55, 20], [50, 10], [45, 0], [50, -10], [55, -15], [60, -10], [65, -5]],
      // 非洲
      [[35, -20], [40, -10], [35, 0], [30, 10], [25, 20], [20, 30], [15, 35], [10, 30], [5, 20], [0, 10], [-5, 0], [-10, -10], [-5, -20], [0, -25], [5, -30], [10, -35], [15, -30], [20, -25], [25, -20], [30, -15]],
      // 北美
      [[70, -170], [75, -160], [80, -150], [85, -140], [80, -130], [75, -120], [70, -110], [65, -100], [60, -90], [55, -80], [50, -70], [45, -60], [40, -50], [35, -60], [30, -70], [25, -80], [20, -90], [15, -100], [10, -110], [5, -120], [0, -130], [-5, -140], [0, -150], [5, -160], [10, -170], [15, -175], [20, -170], [25, -165], [30, -160], [35, -155], [40, -150], [45, -145], [50, -150], [55, -155], [60, -160], [65, -165]],
      // 南美
      [[10, -80], [5, -70], [0, -60], [-5, -50], [-10, -40], [-15, -30], [-20, -20], [-25, -10], [-30, 0], [-35, 10], [-40, 20], [-45, 30], [-50, 40], [-55, 50], [-50, 60], [-45, 70], [-40, 80], [-35, 90], [-30, 100], [-25, 110], [-20, 120], [-15, 130], [-10, 140], [-5, 150], [0, 160], [5, 170], [10, 180], [15, -170], [10, -160], [5, -150], [0, -140], [-5, -130], [-10, -120], [-15, -110], [-20, -100], [-25, -90]],
      // 澳洲
      [[-25, 110], [-30, 120], [-35, 130], [-30, 140], [-25, 150], [-20, 160], [-15, 170], [-10, 180], [-5, -170], [0, -160], [5, -150], [10, -140], [15, -130], [10, -120], [5, -110], [0, -100], [-5, -90], [-10, -80], [-15, -70], [-20, -60], [-25, -50], [-30, -40], [-25, -30], [-20, -20], [-15, -10], [-10, 0], [-5, 10], [0, 20], [5, 30], [10, 40], [15, 50], [20, 60], [25, 70], [30, 80], [35, 90], [40, 100], [35, 110], [30, 120], [25, 130], [20, 140], [15, 150], [10, 160], [5, 170], [0, 180], [-5, -170], [-10, -160], [-15, -150], [-20, -140]]
    ];

    // 绘制大陆填充
    continents.forEach(continent => {
      const points = [];
      continent.forEach(([lat, lng]) => {
        const point3D = latLngTo3D(lat, lng, radius);
        rotatePoint(point3D, rotationRef.current.x, rotationRef.current.y);
        point3D.z += distance;
        const proj = project(point3D, distance);
        if (proj.scale > 0) {
          points.push({ x: centerX + proj.x, y: centerY + proj.y, scale: proj.scale });
        }
      });
      
      if (points.length > 2) {
        // 填充大陆（绿色到棕色渐变，模拟地形）
        const gradient = ctx.createLinearGradient(
          points[0].x, points[0].y,
          points[Math.floor(points.length / 2)].x,
          points[Math.floor(points.length / 2)].y
        );
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.6)'); // 绿色（低地）
        gradient.addColorStop(0.5, 'rgba(101, 163, 13, 0.7)'); // 深绿（平原）
        gradient.addColorStop(1, 'rgba(161, 98, 7, 0.6)'); // 棕色（山地）
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
        ctx.fill();
        
        // 绘制大陆边界
        ctx.strokeStyle = 'rgba(22, 163, 74, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
        ctx.stroke();
      }
    });

    // 绘制当前位置标记
    if (currentPosition) {
      const [lat, lng] = currentPosition;
      const point3D = latLngTo3D(lat, lng, radius);
      rotatePoint(point3D, rotationRef.current.x, rotationRef.current.y);
      point3D.z += distance;
      const proj = project(point3D, distance);
      
      if (proj.scale > 0) {
        // 绘制位置点
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(centerX + proj.x, centerY + proj.y, 4 * proj.scale, 0, Math.PI * 2);
        ctx.fill();
        
        // 绘制外圈
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX + proj.x, centerY + proj.y, 8 * proj.scale, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

  }, [currentPosition]);

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

    // 限制X轴旋转范围
    rotationRef.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationRef.current.x));

    lastMousePosRef.current = { x: currentX, y: currentY };
    drawGlobe();
  }, [drawGlobe]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  // 动画循环
  useEffect(() => {
    if (!isVisible) return;

    const animate = () => {
      drawGlobe();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isVisible, drawGlobe]);

  if (!isVisible) return null;

  return (
    <div className="mini-map-3d-container">
      <div className="mini-map-3d-title">3D地球</div>
      <button className="mini-map-3d-close-btn" onClick={onToggleVisible}>×</button>
      <canvas
        ref={canvasRef}
        width={200}
        height={200}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          cursor: isDraggingRef.current ? 'grabbing' : 'grab'
        }}
      />
    </div>
  );
};

// QQ截图式矩形选择组件（核心）
const QQStyleSelector = ({ onSelectFinish, mapContainerRef, onEnterSelectMode }) => {
  const map = useMapEvents({});
  const maskRef = useRef(null);
  const selectionRef = useRef(null);
  const tipRef = useRef(null);
  const exitBtnRef = useRef(null);
  const startPointRef = useRef(null);
  const isSelectingRef = useRef(false);
  const isSelectModeRef = useRef(false);

  // 退出选择模式
  const exitSelectMode = () => {
    isSelectModeRef.current = false;
    isSelectingRef.current = false;
    if (maskRef.current) maskRef.current.style.display = 'none';
    if (selectionRef.current) selectionRef.current.style.display = 'none';
    if (tipRef.current) tipRef.current.style.display = 'none';
    if (exitBtnRef.current) exitBtnRef.current.style.display = 'none';
    // 启用地图交互
    if (map) {
      map.dragging.enable();
      map.scrollWheelZoom.enable();
      map.doubleClickZoom.enable();
    }
    document.body.style.cursor = 'default';
  };

  // 进入选择模式
  const enterSelectMode = (e) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    console.log('进入选择模式', { 
      maskRef: !!maskRef.current, 
      tipRef: !!tipRef.current, 
      exitBtnRef: !!exitBtnRef.current,
      map: !!map 
    });
    isSelectModeRef.current = true;
    if (maskRef.current) {
      maskRef.current.style.display = 'block';
      console.log('遮罩层已显示');
    }
    if (tipRef.current) {
      tipRef.current.style.display = 'block';
      console.log('提示已显示');
    }
    if (exitBtnRef.current) {
      exitBtnRef.current.style.display = 'block';
      console.log('退出按钮已显示');
    }
    // 禁用地图交互，防止冲突
    if (map) {
      map.dragging.disable();
      map.scrollWheelZoom.disable();
      map.doubleClickZoom.disable();
      console.log('地图交互已禁用');
    }
  };

  // 将enterSelectMode暴露给父组件
  useEffect(() => {
    if (onEnterSelectMode) {
      onEnterSelectMode(enterSelectMode);
    }
  }, [onEnterSelectMode]);

  // 阻止右键菜单
  const handleContextMenu = (e) => {
    if (isSelectModeRef.current) {
      e.preventDefault();
    }
  };

  // 鼠标按下：开始选择
  const handleMouseDown = (e) => {
    if (!isSelectModeRef.current) return;

    // 左键或右键触发选择
    if (e.button === 0 || e.button === 2) {
      e.preventDefault();
      startPointRef.current = { x: e.clientX, y: e.clientY };
      isSelectingRef.current = true;

      // 显示选择框并初始化位置
      if (selectionRef.current) {
        selectionRef.current.style.display = 'block';
        selectionRef.current.style.left = `${e.clientX}px`;
        selectionRef.current.style.top = `${e.clientY}px`;
        selectionRef.current.style.width = '0px';
        selectionRef.current.style.height = '0px';
      }

      document.body.style.cursor = 'crosshair';
    }
  };

  // 鼠标移动：调整选择区域大小
  const handleMouseMove = (e) => {
    if (!isSelectModeRef.current || !isSelectingRef.current || !startPointRef.current || !selectionRef.current) return;

    // 计算选择框位置（确保宽高为正）
    const left = Math.min(startPointRef.current.x, e.clientX);
    const top = Math.min(startPointRef.current.y, e.clientY);
    const width = Math.abs(e.clientX - startPointRef.current.x);
    const height = Math.abs(e.clientY - startPointRef.current.y);

    // 更新选择框样式
    selectionRef.current.style.left = `${left}px`;
    selectionRef.current.style.top = `${top}px`;
    selectionRef.current.style.width = `${width}px`;
    selectionRef.current.style.height = `${height}px`;
  };

  // 鼠标松开：完成选择并跳转
  const handleMouseUp = (e) => {
    if (!isSelectingRef.current || !selectionRef.current) return;

    // 如果是右键松开，也需要阻止默认菜单
    if (e && e.button === 2) {
      e.preventDefault();
    }

    isSelectingRef.current = false;
    document.body.style.cursor = 'default';

    // 获取选择框尺寸（过滤过小区域）
    const { left, top, width, height } = selectionRef.current.getBoundingClientRect();
    if (width < 50 || height < 50) {
      if (tipRef.current) {
        tipRef.current.innerText = '区域过小，请选择更大范围！';
        setTimeout(() => {
          if (tipRef.current) {
            tipRef.current.innerText = '按住左键或右键拖拽选择区域 | 边角可调整大小 | 松开完成选择';
          }
        }, 2000);
      }
      return;
    }

    // 获取地图容器信息，转换为地图坐标
    const mapContainer = mapContainerRef.current;
    if (!mapContainer || !map) return;
    
    const mapRect = mapContainer.getBoundingClientRect();
    
    // 计算选择区域对应的地图经纬度
    const topLeftLatLng = map.containerPointToLatLng([
      left - mapRect.left,
      top - mapRect.top
    ]);
    const bottomRightLatLng = map.containerPointToLatLng([
      left - mapRect.left + width,
      top - mapRect.top + height
    ]);

    // 整理选择数据（传递给目标界面）
    const selectedData = {
      screenRect: { left, top, width, height }, // 屏幕坐标
      mapRect: {
        minLat: topLeftLatLng.lat,
        maxLat: bottomRightLatLng.lat,
        minLng: topLeftLatLng.lng,
        maxLng: bottomRightLatLng.lng,
        center: {
          lat: (topLeftLatLng.lat + bottomRightLatLng.lat) / 2,
          lng: (topLeftLatLng.lng + bottomRightLatLng.lng) / 2
        }
      }
    };

    // 退出选择模式，跳转到目标界面
    exitSelectMode();
    onSelectFinish(selectedData); // 触发跳转
  };

  // 初始化选择相关DOM
  useEffect(() => {
    // 创建遮罩层
    if (!maskRef.current) {
      const mask = document.createElement('div');
      mask.className = 'select-mask';
      document.body.appendChild(mask);
      maskRef.current = mask;
    }

    // 创建矩形选择框（QQ截图风格）
    if (!selectionRef.current) {
      const selection = document.createElement('div');
      selection.className = 'select-selection';
      // 添加4个边角调整点
      ['tl', 'tr', 'bl', 'br'].forEach(type => {
        const handle = document.createElement('div');
        handle.className = `selection-handle selection-handle-${type}`;
        selection.appendChild(handle);
      });
      document.body.appendChild(selection);
      selectionRef.current = selection;
    }

    // 创建操作提示
    if (!tipRef.current) {
      const tip = document.createElement('div');
      tip.className = 'select-tip';
      tip.innerText = '按住左键或右键拖拽选择区域 | 边角可调整大小 | 松开完成选择';
      document.body.appendChild(tip);
      tipRef.current = tip;
    }

    // 创建退出选择按钮
    if (!exitBtnRef.current) {
      const exitBtn = document.createElement('button');
      exitBtn.className = 'exit-select-btn';
      exitBtn.innerText = '退出选择';
      document.body.appendChild(exitBtn);
      exitBtnRef.current = exitBtn;
      exitBtn.addEventListener('click', exitSelectMode);
    }

    // 绑定鼠标事件（核心交互）
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('contextmenu', handleContextMenu);

    return () => {
      // 清理资源
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('contextmenu', handleContextMenu);
      if (exitBtnRef.current) {
        exitBtnRef.current.removeEventListener('click', exitSelectMode);
      }
      // 移除DOM元素
      [maskRef, selectionRef, tipRef, exitBtnRef].forEach(ref => {
        if (ref.current && document.body.contains(ref.current)) {
          document.body.removeChild(ref.current);
        }
      });
    };
  }, [map]);

  // 不渲染按钮，按钮将在App组件中渲染
  return null;
};

const App = () => {
  const mainMapRef = useRef(null); // 地图容器引用
  const [currentPosition, setCurrentPosition] = useState(null);
  const [accuracy, setAccuracy] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isMiniMap3dVisible, setIsMiniMap3dVisible] = useState(true);
  const [isTargetPageVisible, setIsTargetPageVisible] = useState(false); // 控制目标界面显示
  const [selectedData, setSelectedData] = useState(null); // 存储选择的数据
  const [isTyphoonPageVisible, setIsTyphoonPageVisible] = useState(false); // 控制台风界面显示
  const [isTrackingPageVisible, setIsTrackingPageVisible] = useState(false); // 控制追踪界面显示
  const [isTrackingMinimized, setIsTrackingMinimized] = useState(false); // 控制追踪界面是否缩小
  const [selectedTyphoonId, setSelectedTyphoonId] = useState(null); // 当前选中的台风ID
  const [typhoonOpen3D, setTyphoonOpen3D] = useState(false); // 控制打开时是否直接进入3D
  const [globalTime, setGlobalTime] = useState(1); // 全局时间（控制地图上台风的显示）
  const [typhoon1Time, setTyphoon1Time] = useState(1); // 台风1的时间状态
  const [typhoon2Time, setTyphoon2Time] = useState(1); // 台风2的时间状态
  const [typhoon3Time, setTyphoon3Time] = useState(1); // 台风3的时间状态
  const [typhoon1Height, setTyphoon1Height] = useState(1); // 台风1的高度状态
  const [typhoon2Height, setTyphoon2Height] = useState(1); // 台风2的高度状态
  const [typhoon3Height, setTyphoon3Height] = useState(1); // 台风3的高度状态
  const [hoveredTyphoonId, setHoveredTyphoonId] = useState(null); // 当前悬停的台风ID
  const [showHistory, setShowHistory] = useState(false); // 控制历史记录面板显示
  const [visualizationHistory, setVisualizationHistory] = useState([]); // 全局可视化历史记录
  const [showTyphoonOptions, setShowTyphoonOptions] = useState(false); // 控制台风选项弹窗显示
  const [pendingTyphoonId, setPendingTyphoonId] = useState(null); // 待处理的台风ID
  const [showDataExtractor, setShowDataExtractor] = useState(false); // 控制数据提取器面板显示
  const [extractedRegions, setExtractedRegions] = useState([]); // 存储提取的区域信息
  const [typhoonOptions, setTyphoonOptions] = useState({
    open3D: false, // 打开3D视图（PyVista场景）
    useSimulation: true, // 使用模拟数据
    open3DCube: false, // 打开3D转换（运行 text.py 3D视图）
    openCrossSection: false, // 打开取截面
    openVelocity3D: false // 打开3D可视化
  }); // 台风选项

  // 3D小地图切换函数
  const toggleMiniMap3d = () => {
    setIsMiniMap3dVisible(!isMiniMap3dVisible);
  };

  // 定位函数
  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      setError('您的浏览器不支持地理位置功能');
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude, accuracy } = position.coords;
        setCurrentPosition([latitude, longitude]);
        setAccuracy(accuracy);
        setLoading(false);
      },
      (err) => {
        const errorMessages = {
          1: '用户拒绝了定位权限',
          2: '无法获取定位信息',
          3: '定位请求超时'
        };
        setError(errorMessages[err.code] || '定位失败，请重试');
        setLoading(false);
        setCurrentPosition([39.9042, 116.4074]);
        setAccuracy(10000);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  };

  // 初始化定位
  useEffect(() => {
    getCurrentLocation();
  }, []);

  // 手动刷新定位
  const refreshLocation = () => {
    getCurrentLocation();
  };

  // 选择完成：显示目标界面
  const handleSelectFinish = (data) => {
    setSelectedData(data);
    setIsTargetPageVisible(true); // 跳转：显示目标界面
  };

  // 从目标界面返回地图
  const handleBackToMap = () => {
    setIsTargetPageVisible(false);
    setSelectedData(null);
  };

  // 点击台风时显示选项弹窗
  const handleTyphoonClick = async (typhoonId, open3D=false) => {
    setPendingTyphoonId(typhoonId);
    setTyphoonOptions({ 
      open3D, 
      useSimulation: true,
      open3DCube: false,
      openCrossSection: false,
      openVelocity3D: false
    });
    setShowTyphoonOptions(true);
  };

  // 确认选项后跳转到台风界面
  const handleConfirmTyphoonOptions = async () => {
    const typhoonId = pendingTyphoonId;
    if (!typhoonId) return;

    if (typhoonId === 1 || typhoonId === 2) {
      try {
        const timeIndex = getCurrentTyphoonTime() - 1;
        const res = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/api/typhoon?time=${Math.max(0, timeIndex)}&id=${typhoonId}`);
        if (res.ok) {
          const payload = await res.json();
          if (payload && payload.success) {
            const lat = payload.lat;
            const lng = payload.lng;
            if (mainMapRef && mainMapRef.current && mainMapRef.current.setView) {
              try {
                mainMapRef.current.setView([lat, lng], 6);
              } catch (err) {
                // ignore
              }
            }
          }
        }
      } catch (err) {
        console.warn('Failed to fetch typhoon position before opening:', err);
      }
    }
    setSelectedTyphoonId(typhoonId);
    // 将台风的时间状态设置为当前全局时间（进入界面时的时间）
    if (typhoonId === 1) {
      setTyphoon1Time(globalTime);
    } else if (typhoonId === 2) {
      setTyphoon2Time(globalTime);
    } else if (typhoonId === 3) {
      setTyphoon3Time(globalTime);
    }
    // 如果追踪界面打开，则缩小它
    if (isTrackingPageVisible) {
      setIsTrackingMinimized(true);
    }
    setIsTyphoonPageVisible(true);
    setTyphoonOpen3D(typhoonOptions.open3D);
    setShowTyphoonOptions(false);
    setPendingTyphoonId(null);
  };

  // 取消选项弹窗
  const handleCancelTyphoonOptions = () => {
    setShowTyphoonOptions(false);
    setPendingTyphoonId(null);
  };

  // 从台风界面返回地图
  const handleBackFromTyphoon = () => {
    setIsTyphoonPageVisible(false);
    setSelectedTyphoonId(null);
    setTyphoonOpen3D(false);
    // 如果追踪界面被缩小了，恢复它
    if (isTrackingMinimized) {
      setIsTrackingMinimized(false);
    }
  };

  // 全局时间轴切换（控制地图上台风的显示）
  const handleGlobalTimeChange = (time) => {
    setGlobalTime(time);
  };

  // 时间轴切换（针对当前选中的台风）
  const handleTimeChange = (time) => {
    if (selectedTyphoonId === 1) {
      setTyphoon1Time(time);
    } else if (selectedTyphoonId === 2) {
      setTyphoon2Time(time);
    } else if (selectedTyphoonId === 3) {
      setTyphoon3Time(time);
    }
  };

  // 高度轴切换（针对当前选中的台风）
  const handleHeightChange = (height) => {
    if (selectedTyphoonId === 1) {
      setTyphoon1Height(height);
    } else if (selectedTyphoonId === 2) {
      setTyphoon2Height(height);
    } else if (selectedTyphoonId === 3) {
      setTyphoon3Height(height);
    }
  };

  // 获取当前选中台风的时间状态
  const getCurrentTyphoonTime = () => {
    if (selectedTyphoonId === 1) {
      return typhoon1Time;
    } else if (selectedTyphoonId === 2) {
      return typhoon2Time;
    } else if (selectedTyphoonId === 3) {
      return typhoon3Time;
    }
    return 1;
  };

  // 获取当前选中台风的高度状态
  const getCurrentTyphoonHeight = () => {
    if (selectedTyphoonId === 1) {
      return typhoon1Height;
    } else if (selectedTyphoonId === 2) {
      return typhoon2Height;
    } else if (selectedTyphoonId === 3) {
      return typhoon3Height;
    }
    return 1;
  };

  // 处理数据提取成功，在地图上标记区域
  const handleExtractSuccess = useCallback((regionInfo) => {
    // 计算区域中心点（用于调整地图视图）
    const centerLat = (regionInfo.lat_min + regionInfo.lat_max) / 2;
    const centerLon = (regionInfo.lon_min + regionInfo.lon_max) / 2;
    
    // 添加到提取区域列表（直接使用经纬度边界）
    const newRegion = {
      id: Date.now(), // 使用时间戳作为唯一ID
      bounds: [
        [regionInfo.lat_min, regionInfo.lon_min], // 西南角
        [regionInfo.lat_max, regionInfo.lon_max]  // 东北角
      ],
      time_step: regionInfo.time_step,
      summary: regionInfo.summary
    };
    
    setExtractedRegions(prev => [...prev, newRegion]);
    
    // 自动调整地图视图以显示新标记的区域
    if (mainMapRef && mainMapRef.current && mainMapRef.current.setView) {
      try {
        mainMapRef.current.setView([centerLat, centerLon], 6);
      } catch (err) {
        console.warn('Failed to set map view:', err);
      }
    }
  }, []);

  return (
    <>
      <style>{globalStyle}</style>

      <div style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
      }}>
        {/* 顶部标题栏 */}
        <div className="map-header">
          <h1 style={{ margin: 0, fontSize: '1.5rem', color: '#2d3748', fontWeight: 600 }}>
            我的区域地形图
          </h1>
          <div style={{ display: 'flex', gap: 12 }}>
            <button
              style={{
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 16px',
                fontSize: '0.9rem',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#059669'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#10b981'}
              onClick={() => setIsTrackingPageVisible(true)}
              aria-label="台风追踪"
            >
              📍 台风追踪
            </button>
            <button
              style={{
                backgroundColor: '#4299e1',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 16px',
                fontSize: '0.9rem',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#3182ce'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#4299e1'}
              onClick={refreshLocation}
              aria-label="刷新定位"
            >
              🔄 刷新定位
            </button>
            <button
              style={{
                backgroundColor: '#8b5cf6',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 16px',
                fontSize: '0.9rem',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                position: 'relative'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#7c3aed'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#8b5cf6'}
              onClick={() => setShowHistory(true)}
              aria-label="可视化历史"
            >
              📊 可视化历史
              {visualizationHistory.length > 0 && (
                <span style={{
                  position: 'absolute',
                  top: '-6px',
                  right: '-6px',
                  backgroundColor: '#ef4444',
                  color: 'white',
                  borderRadius: '50%',
                  width: '18px',
                  height: '18px',
                  fontSize: '11px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 600
                }}>
                  {visualizationHistory.length}
                </span>
              )}
            </button>
            <button
              style={{
                backgroundColor: '#f59e0b',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 16px',
                fontSize: '0.9rem',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                position: 'relative'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#d97706'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#f59e0b'}
              onClick={() => setShowDataExtractor(true)}
              aria-label="数据提取"
            >
              🌊 数据提取
              {extractedRegions.length > 0 && (
                <span style={{
                  position: 'absolute',
                  top: '-6px',
                  right: '-6px',
                  backgroundColor: '#6b7280',
                  color: 'white',
                  borderRadius: '50%',
                  width: '18px',
                  height: '18px',
                  fontSize: '11px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 600
                }}>
                  {extractedRegions.length}
                </span>
              )}
            </button>
            {extractedRegions.length > 0 && (
              <button
                style={{
                  backgroundColor: '#6b7280',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px 16px',
                  fontSize: '0.9rem',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
                onMouseOver={(e) => e.target.style.backgroundColor = '#4b5563'}
                onMouseOut={(e) => e.target.style.backgroundColor = '#6b7280'}
                onClick={() => setExtractedRegions([])}
                aria-label="清除标记"
                title="清除所有提取区域标记"
              >
                🗑️ 清除标记
              </button>
            )}
          </div>
        </div>

        {/* 主地图（2D地形图） */}
        <MapContainer
          ref={mainMapRef}
          style={{ width: '100%', height: '100%' }}
          center={currentPosition || [39.9042, 116.4074]}
          zoom={13}
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
          <LocationMarker currentPosition={currentPosition} accuracy={accuracy} />
          {/* QQ风格选择组件 */}
          <QQStyleSelector
            onSelectFinish={handleSelectFinish}
            mapContainerRef={mainMapRef}
          />
          {/* 数据提取区域标记（灰色矩形） */}
          {extractedRegions.map((region) => (
            <Rectangle
              key={region.id}
              bounds={region.bounds}
              pathOptions={{
                color: '#374151', // 深灰色边框
                fillColor: '#4b5563', // 深灰色填充
                fillOpacity: 0.3, // 半透明
                weight: 2
              }}
            />
          ))}
        </MapContainer>

        {/* 3D小地图 */}
        <MiniMap3d
          currentPosition={currentPosition}
          isVisible={isMiniMap3dVisible}
          onToggleVisible={toggleMiniMap3d}
        />

        {/* 加载状态 */}
        {loading && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.85)',
            padding: '24px 32px',
            borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
            zIndex: 200,
          }}>
            <div style={{
              width: '40px',
              height: '40px',
              border: '4px solid #e2e8f0',
              borderTop: '4px solid #4299e1',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              marginBottom: '16px',
            }}></div>
            <p style={{ color: '#4a5568', fontSize: '1rem', margin: 0 }}>正在加载地形图并定位...</p>
          </div>
        )}

        {/* 错误提示 */}
        {error && !loading && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            padding: '24px',
            borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
            zIndex: 200,
            maxWidth: '300px',
            textAlign: 'center',
          }}>
            <p style={{ color: '#dc2626', fontSize: '1rem', margin: '0 0 16px 0' }}>{error}</p>
            <button
              style={{
                backgroundColor: '#4299e1',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '10px 20px',
                fontSize: '0.9rem',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#3182ce'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#4299e1'}
              onClick={refreshLocation}
            >
              重试定位
            </button>
          </div>
        )}
      </div>
      {/* 选择完成后跳转的目标界面（默认隐藏，选择后显示） */}
      {isTargetPageVisible && selectedData && (
        <TargetPage
          selectedData={selectedData}
          onBack={handleBackToMap}
        />
      )}

      {/* 主地图时间轴控件 */}
      <TimeAxis currentTime={globalTime} onTimeChange={handleGlobalTimeChange} />

      {/* 台风选项弹窗 */}
      {showTyphoonOptions && pendingTyphoonId && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
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
            <h2 style={{ marginTop: 0, marginBottom: 24, color: '#1f2937' }}>
              台风{pendingTyphoonId} 详情选项
            </h2>
            
            <div style={{ marginBottom: 20 }}>
              <h3 style={{ marginTop: 0, marginBottom: 12, fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                数据选项
              </h3>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: 12 }}>
                <input
                  type="checkbox"
                  checked={typhoonOptions.useSimulation}
                  onChange={(e) => setTyphoonOptions({ ...typhoonOptions, useSimulation: e.target.checked })}
                  style={{ marginRight: 8, width: 18, height: 18, cursor: 'pointer' }}
                />
                <span style={{ fontSize: 15, color: '#374151' }}>
                  使用模拟数据
                </span>
              </label>
            </div>

            <div style={{ marginBottom: 20 }}>
              <h3 style={{ marginTop: 0, marginBottom: 12, fontSize: 16, color: '#4b5563', fontWeight: 600 }}>
                视图选项
              </h3>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: 12 }}>
                <input
                  type="checkbox"
                  checked={typhoonOptions.open3D}
                  onChange={(e) => setTyphoonOptions({ ...typhoonOptions, open3D: e.target.checked })}
                  style={{ marginRight: 8, width: 18, height: 18, cursor: 'pointer' }}
                />
                <span style={{ fontSize: 15, color: '#374151' }}>
                  打开3D视图（PyVista场景）
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: 12 }}>
                <input
                  type="checkbox"
                  checked={typhoonOptions.open3DCube}
                  onChange={(e) => setTyphoonOptions({ ...typhoonOptions, open3DCube: e.target.checked })}
                  style={{ marginRight: 8, width: 18, height: 18, cursor: 'pointer' }}
                />
                <span style={{ fontSize: 15, color: '#374151' }}>
                  {pendingTyphoonId === 3 ? '运行 text.py 3D视图' : '3D转换'}
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: 12 }}>
                <input
                  type="checkbox"
                  checked={typhoonOptions.openCrossSection}
                  onChange={(e) => setTyphoonOptions({ ...typhoonOptions, openCrossSection: e.target.checked })}
                  style={{ marginRight: 8, width: 18, height: 18, cursor: 'pointer' }}
                />
                <span style={{ fontSize: 15, color: '#374151' }}>
                  取截面
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: 12 }}>
                <input
                  type="checkbox"
                  checked={typhoonOptions.openVelocity3D}
                  onChange={(e) => setTyphoonOptions({ ...typhoonOptions, openVelocity3D: e.target.checked })}
                  style={{ marginRight: 8, width: 18, height: 18, cursor: 'pointer' }}
                />
                <span style={{ fontSize: 15, color: '#374151' }}>
                  3D可视化
                </span>
              </label>
            </div>

            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', marginTop: 24, paddingTop: 20, borderTop: '1px solid #e5e7eb' }}>
              <button
                onClick={() => {
                  setShowTyphoonOptions(false);
                  setPendingTyphoonId(null);
                }}
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
                onClick={handleConfirmTyphoonOptions}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#3b82f6',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 6,
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: 500
                }}
              >
                确认
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 台风界面（点击台风按钮后显示） */}
      {isTyphoonPageVisible && selectedTyphoonId && (
        <TyphoonPage
          onBack={handleBackFromTyphoon}
          typhoonId={selectedTyphoonId}
          currentTime={getCurrentTyphoonTime()}
          onTimeChange={handleTimeChange}
          currentHeight={getCurrentTyphoonHeight()}
          onHeightChange={handleHeightChange}
          open3D={typhoonOpen3D}
          useSimulation={typhoonOptions.useSimulation}
          onSaveHistory={(item) => {
            setVisualizationHistory(prev => [item, ...prev]);
          }}
        />
      )}

      {/* 全局可视化历史记录面板 */}
      <VisualizationHistoryPanel
        isVisible={showHistory}
        onClose={() => setShowHistory(false)}
        history={visualizationHistory}
        onClearHistory={() => setVisualizationHistory([])}
      />

      {/* 台风追踪界面 */}
      {isTrackingPageVisible && (
        <TyphoonTracking
          onBack={() => {
            setIsTrackingPageVisible(false);
            setIsTrackingMinimized(false);
          }}
          onTyphoonClick={handleTyphoonClick}
          isMinimized={isTrackingMinimized}
          onRestore={() => setIsTrackingMinimized(false)}
        />
      )}

      {/* 数据提取器面板 */}
      <DataExtractorPanel
        visible={showDataExtractor}
        onClose={() => setShowDataExtractor(false)}
        API_BASE_URL={API_BASE_URL}
        onExtractSuccess={handleExtractSuccess}
      />
    </>
  );
};

export default App;