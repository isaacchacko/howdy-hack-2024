"use client"; // Mark the component as a Client Component

import { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import styles from './page.module.css'; // Import CSS module for styling

interface Data {
    time: number[];
    retention: number[];
    slides: string[];
    slide_changes: number[];
}

const Home = () => {
    const [data, setData] = useState<Data | null>(null);
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:5000/api/data');
                const result: Data = await response.json();
                setData(result);
            } catch (error) {
                console.error('Fetch error:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    // Define chartData here, after the data has been fetched
    const chartData = {
        labels: data?.time || [],
        datasets: [
            {
                label: 'Retention',
                data: data?.retention || [],
                borderColor: 'rgba(100, 150, 200, 1)', // Soft blue for the line
                backgroundColor: 'rgba(100, 150, 200, 0.2)', // Soft blue fill
                borderWidth: 2,
                fill: true, // Fill under the line
                pointHoverRadius: 5,
            },
        ],
    };

    const options = {
        scales: {
            x: {
                grid: {
                    drawOnChartArea: false,
                },
            },
            y: {
                beginAtZero: true,
            },
        },
    };

    const getSlideUrl = (index: number) => {
        if (!data) return '';
        const slideIndex = data.slide_changes.findIndex((change, i) => {
            return (i === 0 && index < change) || (i > 0 && index >= data.slide_changes[i - 1] && index < change);
        });
        return slideIndex === -1 ? data.slides[data.slides.length - 1] : data.slides[slideIndex];
    };

    const handleMouseMove = (event: React.MouseEvent) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const totalWidth = rect.width;
        const index = Math.floor((x / totalWidth) * (data?.time.length || 1));

        if (data) {
            setHoveredIndex(index);
        }
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div className={styles.container}>
            <h1 className={styles.title}>Retention vs. Time Graph</h1>
            <div className={styles.graphContainer} onMouseMove={handleMouseMove}>
                {data && (
                    <Line
                        data={chartData}
                        options={options}
                    />
                )}
                {data?.slide_changes.map((change) => (
                    <div
                        key={change}
                        style={{
                            position: 'absolute',
                            left: `${((data.time.indexOf(change) / (data.time.length - 1)) * 100)}%`,
                            height: '100%',
                            borderLeft: '2px solid #d19a6a', // Light brown for slide changes
                            top: 0,
                        }}
                    />
                ))}
            </div>
            {hoveredIndex !== null && data && (
                <div className={styles.imagePopup}>
                    <img
                        src={getSlideUrl(data.time[hoveredIndex])}
                        alt="Slide Preview"
                    />
                </div>
            )}
        </div>
    );
};

export default Home;
