import locust
from locust import HttpUser, task, between

class PropertyAnalyzerUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(1)
    def test_property_search(self):
        self.client.get("/api/property/search", params={
            "address": "Tollbugata 1, Drammen"
        })
    
    @task(2)
    def test_property_analysis(self):
        with open("test_data/floorplan.pdf", "rb") as f:
            self.client.post("/api/property/analyze", 
                files={"file": f},
                data={"address": "Tollbugata 1, Drammen"}
            )
    
    @task(3)
    def test_document_generation(self):
        self.client.post("/api/documents/generate", json={
            "propertyId": "test-property-id",
            "documentTypes": ["building_application", "situation_plan", "floor_plan"]
        })
    
    @task
    def test_3d_model_generation(self):
        self.client.post("/api/property/generate-3d", json={
            "propertyId": "test-property-id"
        })