/**
 * Metaballs with BOTH random spatial variation (influencers)
 * and microphone-driven pulsing. FRAME-RATE INDEPENDENT.
 *
 * Requires Processing 4.x  +  “Sound” library
 * (Sketch → Import Library… → Add Library… → Sound)
 */

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import processing.sound.*;

// ──────────── Audio ────────────
AudioIn   mic;
Amplitude amp;
float     soundLevel = 0;        // smoothed envelope (0‒1)
final float AUDIO_SENSITIVITY = 2; // tweak overall reactivity
final float MIC_SMOOTHING_RATE = 500.0f; // Higher is faster smoothing, adjust for desired responsiveness

// ──────────── Settings ────────────
int   gridSpacing       = 160;
float baseDiameter      = 20;
float threshold         = 1.0;
float maxRadiusMultiplier = 6.0;
float minRadiusMultiplier = 1.6;
int   marchingGridStep    = 12; // keep constant
int   spatialGridCellSize = 300; // keep constant

// Influencer motion
int   numInfluencers             = 5;
// Original: int momentumChangeIntervalFrames = 180; (Assuming 60fps => 3 seconds)
float momentumChangeIntervalSeconds = 3.0f;
// Original: float influencerMaxSpeed = 10.5f; (pixels/frame, assuming 60fps => 10.5 * 60 = 630 pixels/second)
float influencerMaxSpeedPerSecond   = 1630.0f; // pixels per second
float influencerInfluenceDist    = 200;


// ──────────── Derived constants ────────────
final float BASE_RADIUS                 = baseDiameter / 2f;
final float MIN_BASE_RADIUS_SQ          = sq(BASE_RADIUS * minRadiusMultiplier);
final float MAX_BASE_RADIUS_SQ          = sq(BASE_RADIUS * maxRadiusMultiplier);
final float INFLUENCER_INFLUENCE_DIST_SQ= sq(influencerInfluenceDist);
final float FIELD_CUTOFF_DIST_SQ        = sq(influencerInfluenceDist * 1.5f);
final float DISTANCE_SQ_EPSILON         = 0.0001f * 0.0001f;
final float INV_SPATIAL_GRID_CELL_SIZE  = 1.0f / spatialGridCellSize;
final float SPATIAL_GRID_CHECK_RADIUS   = BASE_RADIUS * maxRadiusMultiplier;

// ──────────── Data arrays ────────────
float[] sourceX,  sourceY,  sourceRadiusSq;
float[] influencerX, influencerY, influencerVX, influencerVY; // VX, VY are now in pixels/second
float[][] fieldValues;
List<Integer>[][] spatialSourceGrid;
int[]   sourceLastCheckVersion;

int cols, rows, spatialCols, spatialRows,
    numSources, currentCheckVersion = 0;

float[] batchedVertices;
int   batchedVertexCount = 0;
final int INITIAL_BATCH_CAPACITY_FLOATS = 40000;

// ──────────── Frame Rate Independence ────────────
long  lastFrameTimeNanos = -1;
float deltaTimeSeconds = 0.016f; // Default to ~60fps for the first frame, will be updated
float timeSinceLastMomentumChange = 0.0f;
final float MAX_DELTA_TIME_SECONDS = 0.1f; // Cap delta time to prevent large jumps (100ms)


// ──────────── Telemetry Globals ────────────
long telem_s0_mic_acc = 0;
long telem_s1_influencers_acc = 0;
long telem_s2_radii_acc = 0;       // Includes updateSpatialGrid
long telem_s3_field_acc = 0;
long telem_s4_render_acc = 0;      // renderMetaballs
long telem_background_acc = 0;
long telem_endOfDraw_acc = 0;      // Time from end of renderMetaballs to end of draw()
long telem_totalFrame_acc = 0;
long telem_frameCount = 0; // For telemetry reporting interval
//frameCount // Processing's built-in frameCount still useful for other non-time-critical logic if needed
final int TELEMETRY_REPORT_INTERVAL = 500; // frames
final float NANOS_TO_MS_FACTOR = 1.0f / 1000000.0f;

// Variables to hold timestamps within a single frame for telemetry
long t_startFrame, t_afterBackground, t_afterMic, t_afterInfluencers, t_afterRadii, t_afterField, t_afterRender, t_endFrame;


// ─────────────────────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────────────────────
void setup() {
  fullScreen(P2D);

  // ── Audio
  mic = new AudioIn(this, 0);
  amp = new Amplitude(this);
  amp.input(mic);
  mic.start();

  // ── Source grid
  ArrayList<PVector> initial = new ArrayList<PVector>();
  for (float x = gridSpacing/2f; x < width;  x += gridSpacing)
    for (float y = gridSpacing/2f; y < height; y += gridSpacing)
      initial.add(new PVector(x, y));

  numSources      = initial.size();
  sourceX         = new float[numSources];
  sourceY         = new float[numSources];
  sourceRadiusSq  = new float[numSources];
  sourceLastCheckVersion = new int[numSources];
  for (int i=0;i<numSources;i++){
    sourceX[i] = initial.get(i).x;
    sourceY[i] = initial.get(i).y;
    sourceRadiusSq[i] = MIN_BASE_RADIUS_SQ;
    sourceLastCheckVersion[i] = -1;
  }

  // ── Influencers (velocities are in pixels/second)
  influencerX  = new float[numInfluencers];
  influencerY  = new float[numInfluencers];
  influencerVX = new float[numInfluencers];
  influencerVY = new float[numInfluencers];
  for (int i=0;i<numInfluencers;i++){
    influencerX[i] = random(width);
    influencerY[i] = random(height);
    influencerVX[i] = random(-influencerMaxSpeedPerSecond, influencerMaxSpeedPerSecond);
    influencerVY[i] = random(-influencerMaxSpeedPerSecond, influencerMaxSpeedPerSecond);
  }

  // ── Grids
  cols = floor(width  / (float)marchingGridStep) + 1;
  rows = floor(height / (float)marchingGridStep) + 1;
  fieldValues = new float[cols][rows];

  spatialCols = floor(width  * INV_SPATIAL_GRID_CELL_SIZE) + 1;
  spatialRows = floor(height * INV_SPATIAL_GRID_CELL_SIZE) + 1;
  spatialSourceGrid = new ArrayList[spatialCols][spatialRows];
  for (int i=0;i<spatialCols;i++)
    for (int j=0;j<spatialRows;j++)
      spatialSourceGrid[i][j] = new ArrayList<Integer>(10);
  updateSpatialGrid();

  // ── Vertex batch
  batchedVertices = new float[INITIAL_BATCH_CAPACITY_FLOATS];
  fill(255);
  noStroke();
 
  // Initialize lastFrameTimeNanos for deltaTime calculation
  lastFrameTimeNanos = System.nanoTime();
  timeSinceLastMomentumChange = random(momentumChangeIntervalSeconds); // Randomize initial change time

  println("Setup complete. Telemetry will be reported every " + TELEMETRY_REPORT_INTERVAL + " frames.");
  println("Influencer max speed: " + influencerMaxSpeedPerSecond + " px/s. Momentum change interval: " + momentumChangeIntervalSeconds + " s.");
}

// ─────────────────────────────────────────────────────────────────────
//  DRAW
// ─────────────────────────────────────────────────────────────────────
void draw() {
  long currentFrameTimeNanos = System.nanoTime();
  if (lastFrameTimeNanos == -1) { // Should have been set in setup, but as a fallback
      lastFrameTimeNanos = currentFrameTimeNanos - 16_666_666; // Assume ~60fps for first delta if not set
  }
  deltaTimeSeconds = (currentFrameTimeNanos - lastFrameTimeNanos) / 1_000_000_000.0f;
  lastFrameTimeNanos = currentFrameTimeNanos;

  // Cap deltaTime to prevent extreme jumps or issues with very low FPS
  deltaTimeSeconds = max(0.00001f, min(deltaTimeSeconds, MAX_DELTA_TIME_SECONDS));


  t_startFrame = currentFrameTimeNanos; // For telemetry

  background(0);
  t_afterBackground = System.nanoTime();

  /* ── 0. Microphone loudness (smooth) ── */
  float raw = amp.analyze();
  // Frame-rate independent lerp:
  float lerpAlpha = 1.0f - exp(-deltaTimeSeconds * MIC_SMOOTHING_RATE);
  soundLevel = lerp(soundLevel, raw, lerpAlpha);
 
  float audioGain = map(constrain(soundLevel * AUDIO_SENSITIVITY, 0, 1),
                        0, 1, 1.0f, 3.0f);   // 1‒3× multiplier
  t_afterMic = System.nanoTime();

  /* ── 1. Move influencers ── */
  timeSinceLastMomentumChange += deltaTimeSeconds;
  boolean changeMomentum = false;
  if (timeSinceLastMomentumChange >= momentumChangeIntervalSeconds) {
    changeMomentum = true;
    timeSinceLastMomentumChange -= momentumChangeIntervalSeconds; // Carry over excess time
  }

  for (int i=0;i<numInfluencers;i++){
    if (changeMomentum){
      influencerVX[i] = random(0, influencerMaxSpeedPerSecond);
      // influencerVX[i] = random(-influencerMaxSpeedPerSecond, influencerMaxSpeedPerSecond);
      influencerVY[i] = 0; //random(-influencerMaxSpeedPerSecond, influencerMaxSpeedPerSecond);
    }
    influencerX[i] = (influencerX[i] + influencerVX[i] * deltaTimeSeconds + width) % width;
    influencerY[i] = (influencerY[i] + influencerVY[i] * deltaTimeSeconds + height) % height;
    // Ensure positive result for modulo with negative numbers if width/height were non-final, though here it's fine.
    if (influencerX[i] < 0) influencerX[i] += width;
    if (influencerY[i] < 0) influencerY[i] += height;
  }
  t_afterInfluencers = System.nanoTime();

  /* ── 2. Update radii: influencer pattern *and* audio ── */
  for (int i=0;i<numSources;i++){
    float minInfDistSq = Float.POSITIVE_INFINITY;
    for (int j=0;j<numInfluencers;j++){
      float dSq = wrappedDistSq(sourceX[i],sourceY[i],
                                influencerX[j],influencerY[j],
                                width,height);
      if (dSq < minInfDistSq) minInfDistSq = dSq;
    }
    float influenceRatio = (minInfDistSq < INFLUENCER_INFLUENCE_DIST_SQ)
                         ? 1.0f - (minInfDistSq / INFLUENCER_INFLUENCE_DIST_SQ)
                         : 0.0f;

    float baseRadiusSq =
        lerp(MIN_BASE_RADIUS_SQ, MAX_BASE_RADIUS_SQ, influenceRatio);

    sourceRadiusSq[i] = baseRadiusSq * sq(audioGain); // audio scaling
  }
  updateSpatialGrid(); // radii changed
  t_afterRadii = System.nanoTime();

  /* ── 3. Compute scalar field ── */
  currentCheckVersion = 0;
  for (int j=0;j<rows;j++){
    float wy = j * marchingGridStep;
    int scy  = floor(wy * INV_SPATIAL_GRID_CELL_SIZE);
    int minCY = max(0, scy-1);
    int maxCY = min(spatialRows-1, scy+1);

    for (int i=0;i<cols;i++){
      float wx = i * marchingGridStep;
      fieldValues[i][j] = calcField(wx, wy, minCY, maxCY);
    }
  }
  t_afterField = System.nanoTime();

  /* ── 4. Marching squares & draw ── */
  renderMetaballs();
  t_afterRender = System.nanoTime();
 
  t_endFrame = System.nanoTime(); // End of draw method timestamp

  // Accumulate telemetry data
  telem_background_acc += (t_afterBackground - t_startFrame);
  telem_s0_mic_acc += (t_afterMic - t_afterBackground);
  telem_s1_influencers_acc += (t_afterInfluencers - t_afterMic);
  telem_s2_radii_acc += (t_afterRadii - t_afterInfluencers);
  telem_s3_field_acc += (t_afterField - t_afterRadii);
  telem_s4_render_acc += (t_afterRender - t_afterField);
  telem_endOfDraw_acc += (t_endFrame - t_afterRender);
  telem_totalFrame_acc += (t_endFrame - t_startFrame);

  telem_frameCount++;

  if (telem_frameCount >= TELEMETRY_REPORT_INTERVAL) {
    float avg_fps_actual = (float)telem_frameCount / (telem_totalFrame_acc * NANOS_TO_MS_FACTOR / 1000.0f);

    println("---- Telemetry Report (avg over " + TELEMETRY_REPORT_INTERVAL + " frames | Avg FPS: " + nf(avg_fps_actual, 0, 1) + ") ----");
   
    float bg_ms     = (telem_background_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float s0_ms     = (telem_s0_mic_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float s1_ms     = (telem_s1_influencers_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float s2_ms     = (telem_s2_radii_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float s3_ms     = (telem_s3_field_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float s4_ms     = (telem_s4_render_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float end_draw_ms = (telem_endOfDraw_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
    float total_ms  = (telem_totalFrame_acc * NANOS_TO_MS_FACTOR) / telem_frameCount;
   
    float sum_of_parts_ms = bg_ms + s0_ms + s1_ms + s2_ms + s3_ms + s4_ms + end_draw_ms;
    float avg_deltaTime_ms = deltaTimeSeconds * 1000.0f; // This is the last deltaTime, not average, but indicative

    println(String.format("  Avg DeltaTime    : %8.3f ms (last frame, target for consistency)", avg_deltaTime_ms));
    println(String.format("  Background Clear : %8.3f ms", bg_ms));
    println(String.format("  0. Mic Loudness  : %8.3f ms", s0_ms));
    println(String.format("  1. Influencers   : %8.3f ms", s1_ms));
    println(String.format("  2. Update Radii  : %8.3f ms (incl. Spatial Grid)", s2_ms));
    println(String.format("  3. Scalar Field  : %8.3f ms", s3_ms));
    println(String.format("  4. Render Metaballs: %8.3f ms", s4_ms));
    println(String.format("  End of Draw      : %8.3f ms (overhead after render)", end_draw_ms));
    println(String.format("  ------------------------------------"));
    println(String.format("  Sum of Parts     : %8.3f ms", sum_of_parts_ms));
    println(String.format("  Total draw() Time: %8.3f ms (%.1f FPS this interval)", total_ms, (total_ms > 0 ? 1000.0f/total_ms : 0) ));
    println("--------------------------------------");

    // Reset accumulators
    telem_background_acc = 0;
    telem_s0_mic_acc = 0;
    telem_s1_influencers_acc = 0;
    telem_s2_radii_acc = 0;
    telem_s3_field_acc = 0;
    telem_s4_render_acc = 0;
    telem_endOfDraw_acc = 0;
    telem_totalFrame_acc = 0;
    telem_frameCount = 0;
  }
}

/* ────────────────── Helpers ────────────────── */
// ... (calcField, renderMetaballs, interp, addTri, addQuad, ensureCapacity, updateSpatialGrid, wrappedDistSq remain unchanged) ...
// Ensure helper functions are present from the original sketch. I'll paste them for completeness.

float calcField(float px,float py,int minCY,int maxCY){
  int scx = floor(px * INV_SPATIAL_GRID_CELL_SIZE);
  int minCX = max(0, scx-1);
  int maxCX = min(spatialCols-1, scx+1);

  float sum = 0;
  currentCheckVersion++;

  for (int cx=minCX; cx<=maxCX; cx++)
    for (int cy=minCY; cy<=maxCY; cy++){
      List<Integer> list = spatialSourceGrid[cx][cy];
      for (int k=0;k<list.size();k++){
        int idx = list.get(k);
        if (sourceLastCheckVersion[idx]==currentCheckVersion) continue;
        sourceLastCheckVersion[idx] = currentCheckVersion;

        float dx = px - sourceX[idx];
        float dy = py - sourceY[idx];
        float d2 = dx*dx + dy*dy;

        if (d2 < FIELD_CUTOFF_DIST_SQ && d2 > DISTANCE_SQ_EPSILON)
          sum += sourceRadiusSq[idx] / d2;
        else if (d2 <= DISTANCE_SQ_EPSILON)
          sum += sourceRadiusSq[idx] / DISTANCE_SQ_EPSILON;
      }
    }
  return sum;
}

void renderMetaballs(){
  batchedVertexCount = 0;
  final float step = marchingGridStep;

  for (int i=0;i<cols-1;i++){
    float x = i*step, x1 = x+step;
    for (int j=0;j<rows-1;j++){
      float y = j*step, y1 = y+step;

      float v00=fieldValues[i  ][j  ];
      float v10=fieldValues[i+1][j  ];
      float v11=fieldValues[i+1][j+1];
      float v01=fieldValues[i  ][j+1];

      int s=0;
      if (v00>=threshold) s|=8;
      if (v10>=threshold) s|=4;
      if (v11>=threshold) s|=2;
      if (v01>=threshold) s|=1;
      if (s==0) continue;
      if (s==15){ addQuad(x,y,x1,y,x1,y1,x,y1); continue; }

      float ax=interp(x ,x1,v00,v10), ay=y;
      float bx=x1, by=interp(y ,y1,v10,v11);
      float cx=interp(x ,x1,v01,v11), cy=y1;
      float dx=x , dy=interp(y ,y1,v00,v01);

      switch(s){
        case 1:  addTri(dx,dy,cx,cy,x ,y1); break;
        case 2:  addTri(cx,cy,bx,by,x1,y1); break;
        case 4:  addTri(ax,ay,bx,by,x1,y ); break;
        case 8:  addTri(ax,ay,dx,dy,x ,y ); break;
        case 3:  addQuad(dx,dy,bx,by,x1,y1,x ,y1); break;
        case 6:  addQuad(ax,ay,cx,cy,x1,y1,x1,y ); break;
        case 9:  addQuad(ax,ay,cx,cy,x ,y1,x ,y ); break;
        case 12: addQuad(dx,dy,bx,by,x1,y ,x ,y ); break;
        case 5:  addTri(ax,ay,bx,by,x1,y ); addTri(cx,cy,dx,dy,x ,y1); break;
        case 10: addTri(ax,ay,dx,dy,x ,y ); addTri(bx,by,cx,cy,x1,y1); break;
        case 7:  addTri(ax,ay,dx,dy,x ,y1); addTri(ax,ay,x ,y1,x1,y1); addTri(ax,ay,x1,y1,x1,y ); break;
        case 11: addTri(bx,by,ax,ay,x ,y ); addTri(bx,by,x ,y ,x ,y1); addTri(bx,by,x ,y1,x1,y1); break;
        case 13: addTri(cx,cy,bx,by,x1,y ); addTri(cx,cy,x1,y ,x ,y ); addTri(cx,cy,x ,y ,x ,y1); break;
        case 14: addTri(dx,dy,cx,cy,x1,y1); addTri(dx,dy,x1,y1,x1,y ); addTri(dx,dy,x1,y ,x ,y ); break;
      }
    }
  }

  if (batchedVertexCount>0){
    beginShape(TRIANGLES);
    for (int k=0;k<batchedVertexCount;k+=2)
      vertex(batchedVertices[k], batchedVertices[k+1]);
    endShape();
  }
}

float interp(float p0,float p1,float v0,float v1){
  if (abs(v0-v1)<1e-5) return (p0+p1)*0.5f;
  float t = (threshold-v0)/(v1-v0);
  return constrain(p0 + t*(p1-p0), min(p0,p1), max(p0,p1)); // Constrain between p0 and p1
}


void addTri(float x1,float y1,float x2,float y2,float x3,float y3){
  ensureCapacity(6);
  batchedVertices[batchedVertexCount++]=x1; batchedVertices[batchedVertexCount++]=y1;
  batchedVertices[batchedVertexCount++]=x2; batchedVertices[batchedVertexCount++]=y2;
  batchedVertices[batchedVertexCount++]=x3; batchedVertices[batchedVertexCount++]=y3;
}

void addQuad(float x1,float y1,float x2,float y2,float x3,float y3,float x4,float y4){
  ensureCapacity(12);
  batchedVertices[batchedVertexCount++]=x1; batchedVertices[batchedVertexCount++]=y1;
  batchedVertices[batchedVertexCount++]=x2; batchedVertices[batchedVertexCount++]=y2;
  batchedVertices[batchedVertexCount++]=x3; batchedVertices[batchedVertexCount++]=y3;
  batchedVertices[batchedVertexCount++]=x1; batchedVertices[batchedVertexCount++]=y1;
  batchedVertices[batchedVertexCount++]=x3; batchedVertices[batchedVertexCount++]=y3;
  batchedVertices[batchedVertexCount++]=x4; batchedVertices[batchedVertexCount++]=y4;
}

void ensureCapacity(int add){
  if (batchedVertexCount+add > batchedVertices.length){
    batchedVertices = Arrays.copyOf(batchedVertices,
        max(batchedVertices.length*2, batchedVertexCount+add));
  }
}

void updateSpatialGrid(){
  for (int i=0;i<spatialCols;i++)
    for (int j=0;j<spatialRows;j++)
      spatialSourceGrid[i][j].clear();

  for (int idx=0; idx<numSources; idx++){
    float sx=sourceX[idx], sy=sourceY[idx];
    int minCX = max(0, floor((sx-SPATIAL_GRID_CHECK_RADIUS)*INV_SPATIAL_GRID_CELL_SIZE));
    int maxCX = min(spatialCols-1, floor((sx+SPATIAL_GRID_CHECK_RADIUS)*INV_SPATIAL_GRID_CELL_SIZE));
    int minCY = max(0, floor((sy-SPATIAL_GRID_CHECK_RADIUS)*INV_SPATIAL_GRID_CELL_SIZE));
    int maxCY = min(spatialRows-1, floor((sy+SPATIAL_GRID_CHECK_RADIUS)*INV_SPATIAL_GRID_CELL_SIZE));
    for (int cx=minCX; cx<=maxCX; cx++)
      for (int cy=minCY; cy<=maxCY; cy++)
        spatialSourceGrid[cx][cy].add(idx);
  }
}

float wrappedDistSq(float x1,float y1,float x2,float y2,float w,float h){
  float dx = x1-x2;
  float dy = y1-y2;
  if (abs(dx) > w/2f) dx -= Math.signum(dx)*w;
  if (abs(dy) > h/2f) dy -= Math.signum(dy)*h;
  return dx*dx + dy*dy;
}
