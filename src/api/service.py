"""FastAPI service for CDR system control and monitoring.

This optional REST API provides:
- Pipeline control (start/stop/status)
- Real-time metrics and status monitoring
- Configuration management
- Scenario loading and execution
- Historical data access
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

from ..cdr.pipeline import CDRPipeline
from ..cdr.schemas import ConfigurationSettings, AircraftState, ConflictPrediction
from ..cdr.metrics import MetricsSummary

logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[CDRPipeline] = None
pipeline_task: Optional[asyncio.Task] = None

app = FastAPI(
    title="LLM-BlueSky CDR System",
    description="Conflict Detection and Resolution API",
    version="0.1.0"
)


class PipelineStatus(BaseModel):
    """Pipeline status response model."""
    running: bool
    cycle_count: int
    start_time: Optional[datetime]
    uptime_seconds: Optional[float]
    last_error: Optional[str]


class StartPipelineRequest(BaseModel):
    """Request to start CDR pipeline."""
    ownship_id: str = "OWNSHIP"
    max_cycles: Optional[int] = None
    config_overrides: Dict[str, Any] = {}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM-BlueSky CDR System",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": {
            "pipeline": "/pipeline/*",
            "metrics": "/metrics",
            "config": "/config",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global pipeline
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "api": "operational",
            "pipeline": "stopped" if pipeline is None else ("running" if pipeline.running else "stopped"),
            "bluesky": "unknown",  # TODO: Check BlueSky connection
            "llm": "unknown"       # TODO: Check LLM availability
        }
    }
    
    return health_status


@app.get("/pipeline/status")
async def get_pipeline_status() -> PipelineStatus:
    """Get current pipeline status."""
    global pipeline, pipeline_task
    
    if pipeline is None:
        return PipelineStatus(
            running=False,
            cycle_count=0,
            start_time=None,
            uptime_seconds=None,
            last_error=None
        )
    
    uptime = None
    if hasattr(pipeline, 'start_time') and pipeline.start_time:
        uptime = (datetime.now() - pipeline.start_time).total_seconds()
    
    return PipelineStatus(
        running=pipeline.running,
        cycle_count=pipeline.cycle_count,
        start_time=getattr(pipeline, 'start_time', None),
        uptime_seconds=uptime,
        last_error=getattr(pipeline, 'last_error', None)
    )


@app.post("/pipeline/start")
async def start_pipeline(
    request: StartPipelineRequest,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """Start the CDR pipeline."""
    global pipeline, pipeline_task
    
    if pipeline is not None and pipeline.running:
        raise HTTPException(status_code=400, detail="Pipeline is already running")
    
    try:
        # Create configuration with any overrides
        config = ConfigurationSettings(**request.config_overrides)
        
        # Initialize pipeline
        pipeline = CDRPipeline(config)
        pipeline.start_time = datetime.now()
        
        # Start pipeline in background
        async def run_pipeline():
            try:
                pipeline.run(
                    max_cycles=request.max_cycles,
                    ownship_id=request.ownship_id
                )
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                pipeline.last_error = str(e)
        
        pipeline_task = asyncio.create_task(run_pipeline())
        
        logger.info(f"Started CDR pipeline for ownship {request.ownship_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Pipeline started for ownship {request.ownship_id}",
                "ownship_id": request.ownship_id,
                "max_cycles": request.max_cycles,
                "start_time": pipeline.start_time.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")


@app.post("/pipeline/stop")
async def stop_pipeline() -> JSONResponse:
    """Stop the CDR pipeline."""
    global pipeline, pipeline_task
    
    if pipeline is None or not pipeline.running:
        raise HTTPException(status_code=400, detail="Pipeline is not running")
    
    try:
        # Stop pipeline gracefully
        pipeline.stop()
        
        # Wait for background task to complete
        if pipeline_task:
            await pipeline_task
            pipeline_task = None
        
        logger.info("CDR pipeline stopped")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Pipeline stopped successfully",
                "final_cycle_count": pipeline.cycle_count,
                "stop_time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline: {str(e)}")


@app.get("/metrics")
async def get_metrics() -> MetricsSummary:
    """Get current performance metrics."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not initialized")
    
    try:
        summary = pipeline.metrics.generate_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate metrics: {str(e)}")


@app.get("/config")
async def get_config() -> ConfigurationSettings:
    """Get current configuration."""
    global pipeline
    
    if pipeline is None:
        # Return default configuration
        return ConfigurationSettings()
    
    return pipeline.config


@app.put("/config")
async def update_config(config: ConfigurationSettings) -> JSONResponse:
    """Update configuration (requires pipeline restart)."""
    global pipeline
    
    if pipeline is not None and pipeline.running:
        raise HTTPException(
            status_code=400,
            detail="Cannot update configuration while pipeline is running. Stop pipeline first."
        )
    
    # Validate configuration
    try:
        # This will raise ValidationError if invalid
        validated_config = ConfigurationSettings(**config.dict())
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Configuration validated. Restart pipeline to apply changes.",
                "config": validated_config.dict()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


@app.get("/aircraft")
async def get_aircraft_states() -> List[AircraftState]:
    """Get current aircraft states from BlueSky."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not initialized")
    
    try:
        aircraft_states = pipeline.bluesky_client.get_aircraft_states()
        return aircraft_states
        
    except Exception as e:
        logger.error(f"Failed to fetch aircraft states: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch aircraft states: {str(e)}")


@app.get("/conflicts")
async def get_current_conflicts() -> List[ConflictPrediction]:
    """Get currently detected conflicts."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not initialized")
    
    # Return recent conflicts from pipeline history
    return pipeline.conflict_history[-10:]  # Last 10 conflicts


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start API server
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
