import React, { useState } from 'react';
import styled from '@emotion/styled';
import { motion, AnimatePresence } from 'framer-motion';
import { FaRobot, FaBrain, FaChartLine, FaBuilding, FaSpinner } from 'react-icons/fa';

const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3c 100%);
  color: white;
  padding: 2rem;
`;

const AnalysisHeader = styled(motion.div)`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled.h1`
  font-size: 3rem;
  background: linear-gradient(120deg, #00ff88, #00a1ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
`;

const ProcessSteps = styled.div`
  display: flex;
  justify-content: space-between;
  max-width: 800px;
  margin: 0 auto 3rem;
`;

const Step = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;

  svg {
    font-size: 2rem;
    color: ${props => props.active ? '#00ff88' : 'rgba(255, 255, 255, 0.5)'};
  }

  p {
    margin: 0;
    opacity: ${props => props.active ? 1 : 0.5};
  }
`;

const ProgressBar = styled(motion.div)`
  width: 100%;
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  margin-bottom: 3rem;
  overflow: hidden;
`;

const Progress = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #00ff88, #00a1ff);
  width: ${props => props.progress}%;
`;

const AnalysisContent = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
`;

const AnalysisCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 20px;
  padding: 2rem;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);

  h3 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #00a1ff;
    margin-top: 0;
  }
`;

const AIProgress = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 10px;

  svg {
    color: #00ff88;
    animation: spin 2s linear infinite;
  }

  @keyframes spin {
    100% { transform: rotate(360deg); }
  }
`;

const InsightList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;

  li {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    padding: 0.8rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;

    &::before {
      content: "→";
      color: #00ff88;
    }
  }
`;

const ActionButton = styled(motion.button)`
  background: linear-gradient(120deg, #00ff88, #00a1ff);
  border: none;
  color: white;
  padding: 1rem 2rem;
  border-radius: 10px;
  font-weight: bold;
  cursor: pointer;
  width: 100%;
  margin-top: 1rem;
`;

const AnalysisPage = () => {
  const [progress, setProgress] = useState(45);
  const [activeStep, setActiveStep] = useState(2);

  return (
    <Container>
      <AnalysisHeader
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Title>AI Analyse Pågår</Title>
        
        <ProcessSteps>
          <Step active={activeStep >= 1}>
            <FaRobot />
            <p>Datainnsamling</p>
          </Step>
          <Step active={activeStep >= 2}>
            <FaBrain />
            <p>AI Analyse</p>
          </Step>
          <Step active={activeStep >= 3}>
            <FaChartLine />
            <p>Rapport Generering</p>
          </Step>
          <Step active={activeStep >= 4}>
            <FaBuilding />
            <p>3D Modellering</p>
          </Step>
        </ProcessSteps>

        <ProgressBar>
          <Progress 
            progress={progress}
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1 }}
          />
        </ProgressBar>
      </AnalysisHeader>

      <AnalysisContent>
        <AnalysisCard
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h3><FaRobot /> AI Prosessering</h3>
          <AIProgress>
            <FaSpinner />
            <p>Analyserer byggeforskrifter...</p>
          </AIProgress>
          <InsightList>
            <li>Identifisert potensial for utbygging</li>
            <li>Analyserer lokale reguleringsplaner</li>
            <li>Beregner optimal utnyttelse</li>
          </InsightList>
        </AnalysisCard>

        <AnalysisCard
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3><FaChartLine /> Foreløpige Resultater</h3>
          <InsightList>
            <li>Potensial for 40% verdiøkning</li>
            <li>Mulighet for kjellerleilighet</li>
            <li>Gunstige markedsforhold</li>
          </InsightList>
          <ActionButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Se detaljert analyse
          </ActionButton>
        </AnalysisCard>

        <AnalysisCard
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <h3><FaBuilding /> 3D Modellering</h3>
          <AIProgress>
            <FaSpinner />
            <p>Genererer 3D modell...</p>
          </AIProgress>
          <InsightList>
            <li>Beregner optimal romfordeling</li>
            <li>Analyserer lysforhold</li>
            <li>Vurderer strukturelle muligheter</li>
          </InsightList>
        </AnalysisCard>
      </AnalysisContent>
    </Container>
  );
};

export default AnalysisPage;