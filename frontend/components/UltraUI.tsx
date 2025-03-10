import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  Button, 
  CircularProgress,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  useTheme,
  useMediaQuery,
  Fab,
  IconButton,
  Tooltip,
  Snackbar,
  Alert
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Map as MapIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Search as SearchIcon,
  Add as AddIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { TerravisionEngine } from './TerravisionEngine';
import PropertyService from '../services/PropertyService';
import VisualizationService from '../services/VisualizationService';
import PropertyAnalyzer from './PropertyAnalyzer';

// Styled components
const MainContainer = ({ children }: { children: React.ReactNode }) => (
  <Container maxWidth={false} sx={{ height: '100vh', overflow: 'hidden', p: 0 }}>
    {children}
  </Container>
);

const ToolbarItem = ({ 
  icon, 
  label, 
  onClick 
}: { 
  icon: React.ReactNode, 
  label: string, 
  onClick: () => void 
}) => (
  <Tooltip title={label}>
    <IconButton color="primary" onClick={onClick}>
      {icon}
    </IconButton>
  </Tooltip>
);

// UltraUI hovedkomponent
const UltraUI: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // State
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState<'map' | 'analytics' | 'settings'>('map');
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');
  const [notificationType, setNotificationType] = useState<'success' | 'error' | 'info'>('info');
  const [analyzedProperties, setAnalyzedProperties] = useState<any[]>([]);
  const [selectedPropertyId, setSelectedPropertyId] = useState<string | null>(null);
  
  // Tjenester
  const propertyService = PropertyService.getInstance();
  const visualizationService = VisualizationService.getInstance();
  
  // Håndter navigasjon
  const handleNavigation = (view: 'map' | 'analytics' | 'settings') => {
    setActiveView(view);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };
  
  // Vis notifikasjon
  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setNotificationMessage(message);
    setNotificationType(type);
    setNotificationOpen(true);
  };
  
  // Last eksempeldata for demo
  const loadExampleData = useCallback(async () => {
    setLoading(true);
    try {
      // Simulerer lasting av data
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const exampleProperties = [
        {
          property_id: "prop_1",
          address: "Storgata 45, Oslo",
          building_potential: {
            max_buildable_area: 450,
            max_height: 15,
            max_units: 5,
            optimal_configuration: "5 leiligheter fordelt på 3 etasjer"
          },
          roi_estimate: 0.18
        },
        {
          property_id: "prop_2",
          address: "Industrivegen 12, Trondheim",
          building_potential: {
            max_buildable_area: 1200,
            max_height: 12,
            max_units: 14,
            optimal_configuration: "Kombinert næring og bolig"
          },
          roi_estimate: 0.22
        }
      ];
      
      setAnalyzedProperties(exampleProperties);
      showNotification('Eksempeldata lastet', 'success');
    } catch (error) {
      console.error('Feil ved lasting av eksempeldata:', error);
      showNotification('Kunne ikke laste eksempeldata', 'error');
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Initialiser data ved oppstart
  useEffect(() => {
    loadExampleData();
  }, [loadExampleData]);
  
  // Responsiv oppsett
  useEffect(() => {
    if (isMobile && drawerOpen) {
      setDrawerOpen(false);
    } else if (!isMobile && !drawerOpen) {
      setDrawerOpen(true);
    }
  }, [isMobile]);
  
  // Rendering av navigasjonsskuff
  const renderDrawer = () => (
    <Drawer
      variant={isMobile ? "temporary" : "persistent"}
      open={drawerOpen}
      onClose={() => setDrawerOpen(false)}
      sx={{
        width: 240,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
          background: theme.palette.background.default,
          borderRight: `1px solid ${theme.palette.divider}`
        },
      }}
    >
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" color="primary" fontWeight="bold">
          EiendomsUI
        </Typography>
        {isMobile && (
          <IconButton onClick={() => setDrawerOpen(false)}>
            <CloseIcon />
          </IconButton>
        )}
      </Box>
      <Divider />
      <List>
        <ListItem 
          button 
          selected={activeView === 'map'}
          onClick={() => handleNavigation('map')}
        >
          <ListItemIcon>
            <MapIcon color={activeView === 'map' ? 'primary' : undefined} />
          </ListItemIcon>
          <ListItemText primary="Visualisering" />
        </ListItem>
        <ListItem 
          button 
          selected={activeView === 'analytics'}
          onClick={() => handleNavigation('analytics')}
        >
          <ListItemIcon>
            <AnalyticsIcon color={activeView === 'analytics' ? 'primary' : undefined} />
          </ListItemIcon>
          <ListItemText primary="Analyse" />
        </ListItem>
        <ListItem 
          button 
          selected={activeView === 'settings'}
          onClick={() => handleNavigation('settings')}
        >
          <ListItemIcon>
            <SettingsIcon color={activeView === 'settings' ? 'primary' : undefined} />
          </ListItemIcon>
          <ListItemText primary="Innstillinger" />
        </ListItem>
      </List>
      <Divider />
      <List>
        <ListItem>
          <Typography variant="subtitle2" color="textSecondary">
            Nylig analyserte
          </Typography>
        </ListItem>
        {analyzedProperties.map(property => (
          <ListItem 
            button 
            key={property.property_id}
            selected={selectedPropertyId === property.property_id}
            onClick={() => setSelectedPropertyId(property.property_id)}
            sx={{ pl: 3 }}
          >
            <ListItemText 
              primary={property.address.split(',')[0]} 
              secondary={`ROI: ${(property.roi_estimate * 100).toFixed(1)}%`}
            />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
  
  // Hovedvisning basert på aktiv fane
  const renderMainContent = () => {
    switch (activeView) {
      case 'map':
        return (
          <Box sx={{ height: 'calc(100vh - 64px)', position: 'relative' }}>
            <Box sx={{ 
              position: 'absolute', 
              top: 0, 
              left: 0, 
              right: 0, 
              zIndex: 1, 
              p: 2, 
              display: 'flex', 
              gap: 1 
            }}>
              <ToolbarItem 
                icon={<SearchIcon />} 
                label="Søk etter eiendom" 
                onClick={() => showNotification('Søk-funksjonalitet kommer snart', 'info')} 
              />
              <ToolbarItem 
                icon={<AddIcon />} 
                label="Legg til bygning" 
                onClick={() => showNotification('Legg til bygning-funksjonalitet kommer snart', 'info')} 
              />
              {/* Legg til flere verktøy her */}
            </Box>
            
            {/* TerravisionEngine-komponent for 3D-visualisering */}
            <div id="terravision-container" style={{ width: '100%', height: '100%' }}>
              {/* TerravisionEngine vil renderes her via JS */}
              <Box sx={{ 
                position: 'absolute', 
                top: '50%', 
                left: '50%', 
                transform: 'translate(-50%, -50%)', 
                textAlign: 'center',
                color: 'text.secondary'
              }}>
                <Typography variant="h6">
                  3D Visualisering lastes...
                </Typography>
                <Typography variant="body2">
                  Her vil TerravisionEngine rendere 3D-modellen
                </Typography>
              </Box>
            </div>
          </Box>
        );
        
      case 'analytics':
        return (
          <Box sx={{ p: 3, height: 'calc(100vh - 64px)', overflow: 'auto' }}>
            <Typography variant="h5" gutterBottom>
              Eiendomsanalyse
            </Typography>
            
            <PropertyAnalyzer />
            
            <Grid container spacing={3} sx={{ mt: 2 }}>
              {analyzedProperties.map(property => (
                <Grid item xs={12} md={6} key={property.property_id}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6">{property.address}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Bygningspotensiale: {property.building_potential.max_buildable_area} m²
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Optimalt: {property.building_potential.optimal_configuration}
                    </Typography>
                    <Typography variant="h6" color="primary" sx={{ mt: 1 }}>
                      ROI: {(property.roi_estimate * 100).toFixed(1)}%
                    </Typography>
                    <Button variant="outlined" size="small" sx={{ mt: 1 }}>
                      Se detaljer
                    </Button>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Box>
        );
        
      case 'settings':
        return (
          <Box sx={{ p: 3, height: 'calc(100vh - 64px)', overflow: 'auto' }}>
            <Typography variant="h5" gutterBottom>
              Innstillinger
            </Typography>
            
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Visualiseringsinnstillinger
              </Typography>
              <Button variant="outlined" size="small">
                Høy kvalitet
              </Button>
              <Button variant="outlined" size="small" sx={{ ml: 1 }}>
                Middels kvalitet
              </Button>
              <Button variant="outlined" size="small" sx={{ ml: 1 }}>
                Lav kvalitet
              </Button>
            </Paper>
            
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                API-tilkobling
              </Typography>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => showNotification('API-test vellykket!', 'success')}
              >
                Test API-tilkobling
              </Button>
            </Paper>
          </Box>
        );
    }
  };
  
  return (
    <MainContainer>
      {/* Toppbar for mobil */}
      {isMobile && (
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          height: 64, 
          px: 2,
          borderBottom: `1px solid ${theme.palette.divider}`
        }}>
          <IconButton onClick={() => setDrawerOpen(true)}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ ml: 2 }}>
            {activeView === 'map' ? 'Visualisering' : 
             activeView === 'analytics' ? 'Analyse' : 'Innstillinger'}
          </Typography>
        </Box>
      )}
      
      {/* Hovedlayout */}
      <Box sx={{ display: 'flex', height: isMobile ? 'calc(100vh - 64px)' : '100vh' }}>
        {renderDrawer()}
        
        <Box sx={{ 
          flexGrow: 1, 
          marginLeft: (!isMobile && drawerOpen) ? '240px' : 0,
          transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}>
          {renderMainContent()}
        </Box>
      </Box>
      
      {/* Floating Action Button på mobil */}
      {isMobile && (
        <Fab 
          color="primary" 
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={() => showNotification('Ny analyse-funksjonalitet kommer snart', 'info')}
        >
          <AddIcon />
        </Fab>
      )}
      
      {/* Laster-indikator */}
      {loading && (
        <Box sx={{ 
          position: 'fixed', 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          zIndex: 9999
        }}>
          <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CircularProgress />
            <Typography sx={{ mt: 2 }}>Laster...</Typography>
          </Paper>
        </Box>
      )}
      
      {/* Notifikasjon */}
      <Snackbar 
        open={notificationOpen} 
        autoHideDuration={6000} 
        onClose={() => setNotificationOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setNotificationOpen(false)} 
          severity={notificationType}
          sx={{ width: '100%' }}
        >
          {notificationMessage}
        </Alert>
      </Snackbar>
    </MainContainer>
  );
};

export default UltraUI; 