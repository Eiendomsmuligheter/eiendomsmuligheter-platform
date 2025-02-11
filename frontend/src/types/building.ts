export interface BuildingLayer {
  id: string;
  name: string;
  description: string;
  visible: boolean;
}

export interface LayerVisibility {
  [key: string]: boolean;
}

export interface Building3DModel {
  url: string;
  format: '3d' | 'floorplan' | 'facade';
  layers: BuildingLayer[];
}

export interface BuildingDimensions {
  width: number;
  length: number;
  height: number;
  area: number;
  volume: number;
}

export interface BuildingRoom {
  id: string;
  name: string;
  area: number;
  dimensions: {
    width: number;
    length: number;
    height: number;
  };
  windows: number;
  doors: number;
}

export interface BuildingFloor {
  id: string;
  name: string;
  level: number;
  rooms: BuildingRoom[];
  area: number;
}

export interface BuildingStructure {
  floors: BuildingFloor[];
  dimensions: BuildingDimensions;
  totalArea: number;
  yearBuilt: number;
}

export interface BuildingUtilities {
  electrical: {
    mainPanel: {
      location: string;
      capacity: number;
    };
    circuits: {
      id: string;
      description: string;
      amperage: number;
    }[];
  };
  plumbing: {
    mainValve: {
      location: string;
    };
    fixtures: {
      id: string;
      type: string;
      location: string;
    }[];
  };
  ventilation: {
    type: string;
    units: {
      id: string;
      location: string;
      capacity: number;
    }[];
  };
}

export interface BuildingRegulations {
  zoning: string;
  maxHeight: number;
  maxFloors: number;
  maxBuildingArea: number;
  setbacks: {
    front: number;
    back: number;
    sides: number;
  };
  parkingRequirements: {
    min: number;
    current: number;
  };
}

export interface CompleteBuilding {
  id: string;
  address: string;
  structure: BuildingStructure;
  utilities: BuildingUtilities;
  regulations: BuildingRegulations;
  model: Building3DModel;
}