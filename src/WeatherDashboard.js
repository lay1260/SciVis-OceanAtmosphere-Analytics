import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import { Thermometer, Droplets, Wind, Gauge, Home } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

function WeatherDashboard({ onBack }) {
  const [weatherData, setWeatherData] = useState([]);
  const [selectedCity, setSelectedCity] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // 模拟天气数据
  useEffect(() => {
    const mockWeatherData = [
      {
        id: 1,
        city: '北京',
        position: [39.9042, 116.4074],
        temperature: 25.6,
        humidity: 65,
        windSpeed: 3.2,
        pressure: 1013,
        condition: 'sunny',
        precipitation: 0
      },
      {
        id: 2,
        city: '上海',
        position: [31.2304, 121.4737],
        temperature: 28.3,
        humidity: 78,
        windSpeed: 4.5,
        pressure: 1008,
        condition: 'cloudy',
        precipitation: 0.2
      },
      {
        id: 3,
        city: '广州',
        position: [23.1291, 113.2644],
        temperature: 32.1,
        humidity: 82,
        windSpeed: 2.8,
        pressure: 1005,
        condition: 'rainy',
        precipitation: 5.4
      },
      {
        id: 4,
        city: '哈尔滨',
        position: [45.756, 126.642],
        temperature: -8.5,
        humidity: 45,
        windSpeed: 5.2,
        pressure: 1020,
        condition: 'snowy',
        precipitation: 2.1
      }
    ];

    setTimeout(() => {
      setWeatherData(mockWeatherData);
      setIsLoading(false);
    }, 1000);
  }, []);

  const getTemperatureColor = (temp) => {
    if (temp < 0) return '#3498db';
    if (temp < 15) return '#2ecc71';
    if (temp < 25) return '#f1c40f';
    if (temp < 35) return '#e67e22';
    return '#e74c3c';
  };

  const getWeatherIcon = (condition) => {
    switch(condition) {
      case 'sunny': return '☀️';
      case 'cloudy': return '☁️';
      case 'rainy': return '🌧️';
      case 'snowy': return '❄️';
      default: return '🌤️';
    }
  };

  if (isLoading) {
    return (
      <div style={{ 
        height: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: '#f5f7fa'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⏳</div>
          <p>正在加载天气数据...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 头部导航 */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '1rem 2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem' }}>实时天气监测仪表盘</h1>
        <button
          onClick={onBack}
          style={{
            background: 'rgba(255, 255, 255, 0.2)',
            border: 'none',
            color: 'white',
            padding: '0.5rem 1rem',
            borderRadius: '5px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}
        >
          <Home size={16} />
          返回首页
        </button>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* 侧边栏 */}
        <div style={{
          width: '300px',
          background: 'white',
          padding: '1.5rem',
          overflowY: 'auto',
          boxShadow: '2px 0 10px rgba(0,0,0,0.1)'
        }}>
          {selectedCity ? (
            <div>
              <h2 style={{ marginBottom: '1.5rem', color: '#2c3e50' }}>
                {selectedCity.city} 天气详情
              </h2>
              
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: '1fr 1fr', 
                gap: '1rem',
                marginBottom: '1.5rem'
              }}>
                <div style={{
                  background: '#f8f9fa',
                  padding: '1rem',
                  borderRadius: '8px',
                  textAlign: 'center',
                  borderLeft: '4px solid #667eea'
                }}>
                  <Thermometer style={{ width: '20px', height: '20px', color: '#667eea', marginBottom: '0.5rem' }} />
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#2c3e50' }}>
                    {selectedCity.temperature}°C
                  </div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.8rem' }}>温度</div>
                </div>
                
                <div style={{
                  background: '#f8f9fa',
                  padding: '1rem',
                  borderRadius: '8px',
                  textAlign: 'center',
                  borderLeft: '4px solid #27ae60'
                }}>
                  <Droplets style={{ width: '20px', height: '20px', color: '#27ae60', marginBottom: '0.5rem' }} />
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#2c3e50' }}>
                    {selectedCity.humidity}%
                  </div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.8rem' }}>湿度</div>
                </div>
                
                <div style={{
                  background: '#f8f9fa',
                  padding: '1rem',
                  borderRadius: '8px',
                  textAlign: 'center',
                  borderLeft: '4px solid '
                }}>
                  <Wind style={{ width: '20px', height: '20px', color: '#e74c3c', marginBottom: '0.5rem' }} />
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#2c3e50' }}>
                    {selectedCity.windSpeed}m/s
                  </div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.8rem' }}>风速</div>
                </div>
                
                <div style={{
                  background: '#f8f9fa',
                  padding: '1rem',
                  borderRadius: '8px',
                  textAlign: 'center',
                  borderLeft: '4px solid '
                }}>
                  <Gauge style={{ width: '20px', height: '20px', color: '#9b59b6', marginBottom: '0.5rem' }} />
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#2c3e50' }}>
                    {selectedCity.pressure}hPa
                  </div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.8rem' }}>气压</div>
                </div>
              </div>

              <div style={{ 
                background: '#f8f9fa', 
                padding: '2rem', 
                borderRadius: '8px', 
                textAlign: 'center',
                color: '#bdc3c7',
                border: '2px dashed #e1e8ed'
              }}>
                温度变化趋势图
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: '#7f8c8d', marginTop: '2rem' }}>
              <h3>选择地图上的城市查看详情</h3>
              <p>点击地图中的天气标记点查看该城市的详细天气信息</p>
            </div>
          )}
        </div>

        {/* 地图区域 */}
        <div style={{ flex: 1, position: 'relative' }}>
          <MapContainer
            center={[35, 105]}
            zoom={4}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {weatherData.map(city => (
              <CircleMarker
                key={city.id}
                center={city.position}
                radius={15}
                fillColor={getTemperatureColor(city.temperature)}
                color="#fff"
                weight={2}
                fillOpacity={0.8}
                eventHandlers={{
                  click: () => setSelectedCity(city)
                }}
              >
                <Popup>
                  <div style={{ textAlign: 'center', minWidth: '120px' }}>
                    <h4 style={{ margin: '0 0 0.5rem 0' }}>{city.city}</h4>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', margin: '0.5rem 0' }}>
                      <span style={{ fontSize: '1.5rem' }}>
                        {getWeatherIcon(city.condition)}
                      </span>
                      <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                        {city.temperature}°C
                      </span>
                    </div>
                    <p style={{ margin: '0.2rem 0', fontSize: '0.9rem' }}>湿度: {city.humidity}%</p>
                    <p style={{ margin: '0.2rem 0', fontSize: '0.9rem' }}>风速: {city.windSpeed}m/s</p>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>
      </div>

      {/* 底部状态栏 */}
      <div style={{
        background: 'white',
        padding: '0.75rem 2rem',
        borderTop: '1px solid #e1e8ed',
        display: 'flex',
        gap: '2rem',
        fontSize: '0.9rem',
        color: '#7f8c8d'
      }}>
        <span>🟢 数据更新: {new Date().toLocaleTimeString()}</span>
        <span>🌡️ 温度范围: -10°C 至 38°C</span>
        <span>⚠️ 预警: 无</span>
      </div>
    </div>
  );
}

export default WeatherDashboard;