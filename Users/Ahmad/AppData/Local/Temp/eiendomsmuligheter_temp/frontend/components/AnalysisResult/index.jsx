import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import { Line, Bar } from 'react-chartjs-2';
import { FaChartLine, FaHome, FaMoneyBillWave, FaClipboardCheck } from 'react-icons/fa';

const Container = styled(motion.div)`
  padding: 2rem;
  background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3c 100%);
  border-radius: 25px;
  color: white;
  margin: 2rem 0;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
`;

const Title = styled.h2`
  font-size: 2.5rem;
  background: linear-gradient(120deg, #00ff88, #00a1ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
`;

const Score = styled.div`
  background: rgba(0, 255, 136, 0.1);
  padding: 1rem 2rem;
  border-radius: 15px;
  border: 1px solid rgba(0, 255, 136, 0.2);
  
  h3 {
    color: #00ff88;
    margin: 0;
    font-size: 2rem;
  }
  
  p {
    margin: 0;
    opacity: 0.7;
  }
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
`;

const Card = styled(motion.div)`
  background: rgba(255, 255, 255, 0.05);
  padding: 1.5rem;
  border-radius: 20px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  
  h3 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0;
    color: #00a1ff;
  }
`;

const ChartContainer = styled.div`
  background: rgba(0, 0, 0, 0.2);
  padding: 1.5rem;
  border-radius: 20px;
  margin-top: 2rem;
`;

const Button = styled(motion.button)`
  background: linear-gradient(120deg, #00ff88, #00a1ff);
  border: none;
  color: white;
  padding: 1rem 2rem;
  border-radius: 10px;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
  }
`;

const FeatureList = styled.ul`
  list-style: none;
  padding: 0;
  
  li {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    
    &::before {
      content: "✓";
      color: #00ff88;
    }
  }
`;

const AnalysisResult = ({ data }) => {
  // Example data for charts
  const chartData = {
    labels: ['2020', '2021', '2022', '2023', '2024', '2025'],
    datasets: [
      {
        label: 'Verdiutvikling',
        data: [300, 350, 400, 480, 520, 580],
        borderColor: '#00ff88',
        tension: 0.4,
      },
    ],
  };

  return (
    <Container
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <Header>
        <Title>Eiendomsanalyse</Title>
        <Score>
          <h3>92/100</h3>
          <p>Utviklingspotensial</p>
        </Score>
      </Header>

      <Grid>
        <Card
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <h3>
            <FaHome />
            Utviklingsmuligheter
          </h3>
          <FeatureList>
            <li>Mulighet for kjellerleilighet</li>
            <li>Potensial for påbygg</li>
            <li>Garasje kan konverteres</li>
            <li>Mulighet for takterrasse</li>
          </FeatureList>
        </Card>

        <Card
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <h3>
            <FaMoneyBillWave />
            Økonomisk Analyse
          </h3>
          <FeatureList>
            <li>Estimert ROI: 15-20%</li>
            <li>Byggekostnader: 2.5M NOK</li>
            <li>Forventet verdistigning: 30%</li>
            <li>Leieinntektspotensial: 25000/mnd</li>
          </FeatureList>
        </Card>

        <Card
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <h3>
            <FaClipboardCheck />
            Reguleringsanalyse
          </h3>
          <FeatureList>
            <li>Reguleringsplan godkjenner utbygging</li>
            <li>Ingen vernestatus</li>
            <li>God utnyttelsesgrad</li>
            <li>Enkel byggesøknadsprosess</li>
          </FeatureList>
        </Card>
      </Grid>

      <ChartContainer>
        <h3>
          <FaChartLine />
          Verdiutvikling og Prognoser
        </h3>
        <Line data={chartData} options={{
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              }
            },
            x: {
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              }
            }
          }
        }} />
      </ChartContainer>

      <motion.div
        style={{ textAlign: 'center', marginTop: '2rem' }}
        whileHover={{ scale: 1.05 }}
      >
        <Button>
          Last ned fullstendig rapport
        </Button>
      </motion.div>
    </Container>
  );
};

export default AnalysisResult;