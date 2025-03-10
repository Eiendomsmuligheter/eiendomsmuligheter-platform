import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Grid, 
  Typography, 
  Paper, 
  Divider, 
  List, 
  ListItem, 
  ListItemIcon,
  ListItemText,
  Chip,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  styled,
  useTheme
} from '@mui/material';
import {
  Business as BusinessIcon,
  Apartment as ApartmentIcon,
  Architecture as ArchitectureIcon,
  Bolt as BoltIcon,
  AttachMoney as MoneyIcon,
  ArrowUpward as PositiveIcon,
  ArrowDownward as NegativeIcon,
  ErrorOutline as RiskIcon,
  Recommend as RecommendIcon,
  Add as PlusIcon,
  Warning as WarningIcon,
  Check as CheckIcon,
  PieChart as ChartIcon,
  Eco as EcoIcon,
  Home as HomeIcon
} from '@mui/icons-material';

// Import TerravisionEngine kun hvis i en browser
let TerravisionEngine = null;
if (typeof window !== 'undefined') {
  import('../components/TerrainVisualizer').then(module => {
    TerravisionEngine = module.default;
  }).catch(error => {
    console.error('Kunne ikke laste TerrainVisualizer:', error);
  });
}

// TypeScript-grensesnitt
interface RegulationRule {
  id: string;
  rule_type: string;
  value: any;
  description: string;
  unit?: string;
  category?: string;
}

interface BuildingPotential {
  max_buildable_area: number;
  max_height: number;
  max_units: number;
  optimal_configuration: string;
  constraints?: string[];
  recommendations?: string[];
}

interface EnergyProfile {
  energy_class: string;
  heating_demand: number;
  cooling_demand: number;
  primary_energy_source: string;
  recommendations?: string[];
}

interface AnalysisResult {
  property_id: string;
  address: string;
  regulations: RegulationRule[];
  building_potential: BuildingPotential;
  energy_profile?: EnergyProfile;
  roi_estimate?: number;
  risk_assessment?: Record<string, string>;
  recommendations?: string[];
}

// Stilte komponenter
const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: theme.spacing(1),
  height: '100%',
  position: 'relative',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: theme.shadows[6]
  }
}));

const VisualizationContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  height: 400,
  backgroundColor: '#f5f5f5',
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
  marginBottom: theme.spacing(3)
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontSize: '1.25rem',
  fontWeight: 600,
  marginBottom: theme.spacing(2),
  display: 'flex',
  alignItems: 'center',
  '& svg': {
    marginRight: theme.spacing(1)
  }
}));

const ValueChip = styled(Chip)<{ strength?: 'positive' | 'negative' | 'neutral' }>(
  ({ theme, strength = 'neutral' }) => ({
    fontWeight: 600,
    backgroundColor: 
      strength === 'positive' 
        ? theme.palette.success.light 
        : strength === 'negative' 
          ? theme.palette.error.light 
          : theme.palette.grey[200],
    color: 
      strength === 'positive' 
        ? theme.palette.success.contrastText 
        : strength === 'negative' 
          ? theme.palette.error.contrastText 
          : theme.palette.text.primary
  })
);

const TabPanel = (props: { children?: React.ReactNode; index: number; value: number }) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
};

// Hovedkomponenten
const PropertyAnalysisViewer: React.FC<{ 
  analysisData: AnalysisResult | null;
  isLoading?: boolean;
  error?: string;
  onGenerateReport?: () => void;
  onVisualize3D?: () => void;
}> = ({ analysisData, isLoading = false, error, onGenerateReport, onVisualize3D }) => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const visualizationRef = useRef<HTMLDivElement>(null);
  const [visualization, setVisualization] = useState<any>(null);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Initialiser 3D-visualisering når data er tilgjengelig
  useEffect(() => {
    if (
      !isLoading && 
      analysisData && 
      visualizationRef.current && 
      typeof window !== 'undefined' && 
      TerravisionEngine
    ) {
      // Rydd opp eksisterende
      if (visualization) {
        visualization.dispose();
      }
      
      // TODO: Implementer faktisk visualisering med TerravisionEngine
      // Dette er en placeholder for fremtidig implementasjon
    }
    
    // Rydd opp ved unmount
    return () => {
      if (visualization) {
        visualization.dispose();
      }
    };
  }, [analysisData, isLoading]);
  
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
        <Typography variant="body1" sx={{ ml: 2 }}>
          Analyserer eiendom...
        </Typography>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <WarningIcon color="error" fontSize="large" />
        <Typography variant="body1" color="error" sx={{ ml: 2 }}>
          {error}
        </Typography>
      </Box>
    );
  }
  
  if (!analysisData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <Typography variant="body1" color="textSecondary">
          Ingen analysedata tilgjengelig. Vennligst velg en eiendom for å starte analysen.
        </Typography>
      </Box>
    );
  }
  
  // Formater ROI til prosent
  const formattedROI = analysisData.roi_estimate 
    ? `${(analysisData.roi_estimate * 100).toFixed(1)}%` 
    : 'Ikke tilgjengelig';
    
  // Bestem ROI-styrke
  const roiStrength = analysisData.roi_estimate
    ? analysisData.roi_estimate > 0.15 
      ? 'positive' 
      : analysisData.roi_estimate < 0.05 
        ? 'negative' 
        : 'neutral'
    : 'neutral';
  
  // Kategoriser reguleringer
  const regulationsByCategory = analysisData.regulations.reduce((acc, regulation) => {
    const category = regulation.category || 'other';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(regulation);
    return acc;
  }, {} as Record<string, RegulationRule[]>);
  
  return (
    <Box sx={{ width: '100%' }}>
      {/* Toppinformasjon */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          {analysisData.address}
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Eiendoms-ID: {analysisData.property_id}
        </Typography>
      </Box>
      
      {/* Hovedtall */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StyledPaper>
            <Typography variant="subtitle2" color="textSecondary">
              Maksimalt byggbart areal
            </Typography>
            <Typography variant="h4">
              {analysisData.building_potential.max_buildable_area} m²
            </Typography>
          </StyledPaper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StyledPaper>
            <Typography variant="subtitle2" color="textSecondary">
              Antall potensielle enheter
            </Typography>
            <Typography variant="h4">
              {analysisData.building_potential.max_units}
            </Typography>
          </StyledPaper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StyledPaper>
            <Typography variant="subtitle2" color="textSecondary">
              Maksimal bygningshøyde
            </Typography>
            <Typography variant="h4">
              {analysisData.building_potential.max_height} m
            </Typography>
          </StyledPaper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StyledPaper>
            <Typography variant="subtitle2" color="textSecondary">
              Estimert ROI
            </Typography>
            <Typography variant="h4">
              <ValueChip 
                label={formattedROI} 
                strength={roiStrength}
                icon={roiStrength === 'positive' ? <PositiveIcon /> : roiStrength === 'negative' ? <NegativeIcon /> : undefined}
              />
            </Typography>
          </StyledPaper>
        </Grid>
      </Grid>
      
      {/* 3D Visualisering */}
      <VisualizationContainer ref={visualizationRef}>
        <Box 
          sx={{ 
            width: '100%', 
            height: '100%', 
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          <Typography variant="body1" color="textSecondary">
            3D-visualisering vil vises her
          </Typography>
          {onVisualize3D && (
            <Button 
              variant="contained" 
              color="primary" 
              onClick={onVisualize3D}
              startIcon={<BusinessIcon />}
              sx={{ ml: 2 }}
            >
              Åpne 3D-visning
            </Button>
          )}
        </Box>
      </VisualizationContainer>
      
      {/* Tabs for detaljvisning */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
          <Tab label="Byggemuligheter" icon={<ApartmentIcon />} iconPosition="start" />
          <Tab label="Reguleringsbestemmelser" icon={<ArchitectureIcon />} iconPosition="start" />
          <Tab label="Energi & Bærekraft" icon={<BoltIcon />} iconPosition="start" />
          <Tab label="Økonomi & Risiko" icon={<MoneyIcon />} iconPosition="start" />
        </Tabs>
      </Box>
      
      {/* Byggemuligheter Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <SectionTitle>
                <ApartmentIcon color="primary" />
                Optimal Konfigurasjon
              </SectionTitle>
              <Typography variant="h6" gutterBottom>
                {analysisData.building_potential.optimal_configuration}
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <SectionTitle>
                <RecommendIcon color="primary" />
                Anbefalinger
              </SectionTitle>
              <List>
                {analysisData.building_potential.recommendations?.map((recommendation, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckIcon color="success" />
                    </ListItemIcon>
                    <ListItemText primary={recommendation} />
                  </ListItem>
                ))}
              </List>
            </StyledPaper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <SectionTitle>
                <WarningIcon color="primary" />
                Begrensninger
              </SectionTitle>
              <List>
                {analysisData.building_potential.constraints?.map((constraint, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <WarningIcon color="warning" />
                    </ListItemIcon>
                    <ListItemText primary={constraint} />
                  </ListItem>
                ))}
              </List>
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
      
      {/* Reguleringsbestemmelser Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          {Object.entries(regulationsByCategory).map(([category, rules], index) => (
            <Grid item xs={12} md={6} key={category}>
              <StyledPaper>
                <SectionTitle>
                  <ArchitectureIcon color="primary" />
                  {category.charAt(0).toUpperCase() + category.slice(1)} Bestemmelser
                </SectionTitle>
                <List>
                  {rules.map((rule) => (
                    <ListItem key={rule.id}>
                      <ListItemText 
                        primary={rule.description} 
                        secondary={`${rule.value}${rule.unit ? ' ' + rule.unit : ''}`}
                      />
                      <Chip 
                        label={rule.rule_type} 
                        size="small" 
                        variant="outlined"
                      />
                    </ListItem>
                  ))}
                </List>
              </StyledPaper>
            </Grid>
          ))}
        </Grid>
      </TabPanel>
      
      {/* Energi & Bærekraft Tab */}
      <TabPanel value={tabValue} index={2}>
        {analysisData.energy_profile ? (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <StyledPaper>
                <SectionTitle>
                  <BoltIcon color="primary" />
                  Energiprofil
                </SectionTitle>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1">Energiklasse</Typography>
                  <Box display="flex" alignItems="center">
                    <Chip 
                      label={analysisData.energy_profile.energy_class}
                      sx={{ 
                        fontWeight: 'bold',
                        fontSize: '1.2rem',
                        backgroundColor: 
                          analysisData.energy_profile.energy_class === 'A' ? '#4CAF50' : 
                          analysisData.energy_profile.energy_class === 'B' ? '#8BC34A' :
                          analysisData.energy_profile.energy_class === 'C' ? '#CDDC39' :
                          analysisData.energy_profile.energy_class === 'D' ? '#FFEB3B' :
                          analysisData.energy_profile.energy_class === 'E' ? '#FFC107' :
                          analysisData.energy_profile.energy_class === 'F' ? '#FF9800' :
                          '#F44336',
                        color: ['A', 'B', 'C'].includes(analysisData.energy_profile.energy_class) ? 'white' : 'black'
                      }}
                    />
                  </Box>
                </Box>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle1">Oppvarmingsbehov</Typography>
                    <Typography variant="body1">
                      {analysisData.energy_profile.heating_demand} kWh/m²
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle1">Kjølebehov</Typography>
                    <Typography variant="body1">
                      {analysisData.energy_profile.cooling_demand} kWh/m²
                    </Typography>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1">Primær energikilde</Typography>
                  <Typography variant="body1">
                    {analysisData.energy_profile.primary_energy_source}
                  </Typography>
                </Box>
              </StyledPaper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <StyledPaper>
                <SectionTitle>
                  <EcoIcon color="primary" />
                  Bærekraftsanbefalinger
                </SectionTitle>
                <List>
                  {analysisData.energy_profile.recommendations?.map((recommendation, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <EcoIcon color="success" />
                      </ListItemIcon>
                      <ListItemText primary={recommendation} />
                    </ListItem>
                  ))}
                </List>
              </StyledPaper>
            </Grid>
          </Grid>
        ) : (
          <Typography variant="body1" color="textSecondary">
            Energiprofil er ikke tilgjengelig for denne eiendommen.
          </Typography>
        )}
      </TabPanel>
      
      {/* Økonomi & Risiko Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <SectionTitle>
                <MoneyIcon color="primary" />
                Økonomisk Potensiale
              </SectionTitle>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1">Return on Investment (ROI)</Typography>
                <Box display="flex" alignItems="center">
                  <Typography variant="h4" sx={{ mr: 1 }}>
                    {formattedROI}
                  </Typography>
                  {roiStrength === 'positive' && <PositiveIcon color="success" />}
                  {roiStrength === 'negative' && <NegativeIcon color="error" />}
                </Box>
              </Box>
              <Divider sx={{ my: 2 }} />
              <Box>
                <Typography variant="subtitle1">Anbefalinger for maksimering av verdi</Typography>
                <List>
                  {analysisData.recommendations?.map((recommendation, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <MoneyIcon color="primary" />
                      </ListItemIcon>
                      <ListItemText primary={recommendation} />
                    </ListItem>
                  ))}
                </List>
              </Box>
            </StyledPaper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <SectionTitle>
                <RiskIcon color="primary" />
                Risikovurdering
              </SectionTitle>
              {analysisData.risk_assessment ? (
                <List>
                  {Object.entries(analysisData.risk_assessment).map(([key, value], index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <RiskIcon 
                          color={
                            value === 'low' 
                              ? 'success' 
                              : value === 'medium' 
                                ? 'warning' 
                                : 'error'
                          } 
                        />
                      </ListItemIcon>
                      <ListItemText 
                        primary={key.split('_').map(word => 
                          word.charAt(0).toUpperCase() + word.slice(1)
                        ).join(' ')} 
                        secondary={value.charAt(0).toUpperCase() + value.slice(1)}
                      />
                      <Chip 
                        label={value.toUpperCase()} 
                        color={
                          value === 'low' 
                            ? 'success' 
                            : value === 'medium' 
                              ? 'warning' 
                              : 'error'
                        }
                        size="small"
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body1" color="textSecondary">
                  Risikovurdering er ikke tilgjengelig for denne eiendommen.
                </Typography>
              )}
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
      
      {/* Rapportgenerering */}
      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        {onGenerateReport && (
          <Button 
            variant="contained" 
            color="primary" 
            onClick={onGenerateReport}
            startIcon={<ChartIcon />}
            size="large"
          >
            Generer fullstendig rapport
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default PropertyAnalysisViewer; 