import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import { FaSearch, FaChartLine, FaRobot, FaBuilding } from 'react-icons/fa';

const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3c 100%);
  color: white;
`;

const Hero = styled.div`
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 0 2rem;
  background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
              url('/images/modern-building.jpg');
  background-size: cover;
  background-position: center;
`;

const Title = styled(motion.h1)`
  font-size: 4rem;
  margin-bottom: 1rem;
  background: linear-gradient(120deg, #00ff88, #00a1ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const Subtitle = styled(motion.p)`
  font-size: 1.5rem;
  max-width: 800px;
  margin-bottom: 2rem;
  opacity: 0.8;
`;

const SearchBar = styled(motion.div)`
  display: flex;
  gap: 1rem;
  margin-bottom: 4rem;
  width: 100%;
  max-width: 600px;
`;

const Input = styled.input`
  flex: 1;
  padding: 1rem 2rem;
  border-radius: 50px;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  color: white;
  font-size: 1.1rem;
  
  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }
`;

const SearchButton = styled(motion.button)`
  padding: 1rem 2rem;
  border-radius: 50px;
  border: none;
  background: linear-gradient(120deg, #00ff88, #00a1ff);
  color: white;
  font-weight: bold;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const Features = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  padding: 4rem 2rem;
  max-width: 1200px;
  margin: 0 auto;
`;

const FeatureCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.05);
  padding: 2rem;
  border-radius: 20px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  text-align: center;
  
  svg {
    font-size: 3rem;
    margin-bottom: 1rem;
    color: #00ff88;
  }
  
  h3 {
    margin-bottom: 1rem;
    color: #00a1ff;
  }
`;

const Home = () => {
  return (
    <Container>
      <Hero>
        <Title
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          Eiendomsmuligheter
        </Title>
        <Subtitle
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Avdekk skjulte muligheter i din eiendom med vår AI-drevne analyseplattform
        </Subtitle>
        
        <SearchBar
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <Input placeholder="Skriv inn adressen til eiendommen" />
          <SearchButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <FaSearch /> Analyser
          </SearchButton>
        </SearchBar>

        <Features>
          <FeatureCard
            whileHover={{ y: -10 }}
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <FaRobot />
            <h3>AI-Drevet Analyse</h3>
            <p>Avansert maskinlæring analyserer alle aspekter av din eiendom</p>
          </FeatureCard>

          <FeatureCard
            whileHover={{ y: -10 }}
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <FaBuilding />
            <h3>3D Visualisering</h3>
            <p>Se potensielle endringer i interaktive 3D-modeller</p>
          </FeatureCard>

          <FeatureCard
            whileHover={{ y: -10 }}
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1 }}
          >
            <FaChartLine />
            <h3>Økonomisk Analyse</h3>
            <p>Detaljerte ROI-beregninger og markedsanalyser</p>
          </FeatureCard>
        </Features>
      </Hero>
    </Container>
  );
};

export default Home;