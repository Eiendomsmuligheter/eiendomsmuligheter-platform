import React from 'react';
import styles from '../styles/AnalysisResult.module.css';

const AnalysisResult = ({ analysis }) => {
    if (!analysis) return null;

    return (
        <div className={styles.analysisContainer}>
            <h2>Analyse av {analysis.address}</h2>
            
            <div className={styles.summarySection}>
                <h3>AI Analyse</h3>
                <p>{analysis.aiAnalysis.recommendation}</p>
                <div className={styles.scoreContainer}>
                    <div>Konfidensscore: {analysis.aiAnalysis.confidenceScore * 100}%</div>
                    <div>Potensielt verdiøkning: {analysis.aiAnalysis.potentialValue} NOK</div>
                </div>
            </div>

            <div className={styles.possibilitiesSection}>
                <h3>Muligheter</h3>
                {analysis.possibilities.map((possibility, index) => (
                    <div key={index} className={styles.possibilityCard}>
                        <h4>{possibility.type}</h4>
                        <p><strong>Gjennomførbarhet:</strong> {possibility.feasibility}</p>
                        <p><strong>Estimert kostnad:</strong> {possibility.estimatedCost} NOK</p>
                        <div className={styles.requirements}>
                            <strong>Krav:</strong>
                            <ul>
                                {possibility.requirements.map((req, i) => (
                                    <li key={i}>{req}</li>
                                ))}
                            </ul>
                        </div>
                        <p>{possibility.description}</p>
                    </div>
                ))}
            </div>

            <div className={styles.propertyData}>
                <h3>Eiendomsinfo</h3>
                <ul>
                    <li>Størrelse: {analysis.propertyData.size} kvm</li>
                    <li>Byggeår: {analysis.propertyData.yearBuilt}</li>
                    <li>Type: {analysis.propertyData.propertyType}</li>
                    <li>Regulering: {analysis.propertyData.zoning}</li>
                </ul>
            </div>
        </div>
    );
};

export default AnalysisResult;