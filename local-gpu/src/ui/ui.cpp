#include "ui.h"
#include "../app.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// Web CSS color reference:
//   --bg:        #01030a        --surface:   rgba(255,255,255,0.04)
//   --border:    rgba(255,255,255,0.07)      --border-hi: rgba(255,255,255,0.13)
//   --accent-a:  #22d3ee        --accent-b:  #0ea5e9
//   --text:      #e2eaf2        --muted:     #4a5f72       --muted2: #6b8096
// Panel bg: rgba(1,6,18,0.78)   border-radius: 18px
// ---------------------------------------------------------------------------

// Color helpers matching web CSS vars
static const ImVec4 COL_TEXT     = ImVec4(0.886f, 0.918f, 0.949f, 1.0f);  // #e2eaf2
static const ImVec4 COL_MUTED   = ImVec4(0.290f, 0.373f, 0.447f, 1.0f);  // #4a5f72
static const ImVec4 COL_MUTED2  = ImVec4(0.420f, 0.502f, 0.588f, 1.0f);  // #6b8096
static const ImVec4 COL_ACCENT  = ImVec4(0.133f, 0.827f, 0.933f, 1.0f);  // #22d3ee
static const ImVec4 COL_SURFACE = ImVec4(1.0f, 1.0f, 1.0f, 0.04f);
static const ImVec4 COL_BORDER  = ImVec4(1.0f, 1.0f, 1.0f, 0.07f);

namespace UI {

void init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiStyle& s = ImGui::GetStyle();
    // Match web: border-radius 18px panel, 9px buttons
    s.WindowRounding    = 18.0f;
    s.FrameRounding     = 9.0f;
    s.GrabRounding      = 6.0f;
    s.WindowPadding     = ImVec2(18, 18);
    s.ItemSpacing       = ImVec2(5, 5);  // gap: 5px in web grid
    s.FramePadding      = ImVec2(0, 9);  // padding: 9px 0 in buttons
    s.WindowBorderSize  = 1.0f;
    s.FrameBorderSize   = 1.0f;

    // Panel bg: rgba(1,6,18,0.78)
    s.Colors[ImGuiCol_WindowBg]         = ImVec4(0.004f, 0.024f, 0.071f, 0.78f);
    s.Colors[ImGuiCol_Border]           = COL_BORDER;
    s.Colors[ImGuiCol_Text]             = COL_TEXT;
    s.Colors[ImGuiCol_TextDisabled]     = COL_MUTED2;

    // Default button: surface bg, border
    s.Colors[ImGuiCol_Button]           = COL_SURFACE;
    s.Colors[ImGuiCol_ButtonHovered]    = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);
    s.Colors[ImGuiCol_ButtonActive]     = ImVec4(1.0f, 1.0f, 1.0f, 0.10f);

    // Frame (unused mostly but set for consistency)
    s.Colors[ImGuiCol_FrameBg]          = COL_SURFACE;
    s.Colors[ImGuiCol_FrameBgHovered]   = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);
    s.Colors[ImGuiCol_FrameBgActive]    = ImVec4(1.0f, 1.0f, 1.0f, 0.10f);

    s.Colors[ImGuiCol_Separator]        = COL_BORDER;
    s.Colors[ImGuiCol_Header]           = COL_SURFACE;
    s.Colors[ImGuiCol_HeaderHovered]    = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);

    // Title bar (hidden, but just in case)
    s.Colors[ImGuiCol_TitleBg]          = ImVec4(0.004f, 0.024f, 0.071f, 0.90f);
    s.Colors[ImGuiCol_TitleBgActive]    = ImVec4(0.004f, 0.024f, 0.071f, 0.90f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
}

// Helper: tidal-style button (web .btn.tidal)
// bg: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(56,189,248,0.08))
// border: rgba(34,211,238,0.25)  color: #67e8f9
static bool tidalButton(const char* label, ImVec2 size) {
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.133f, 0.827f, 0.933f, 0.10f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.133f, 0.827f, 0.933f, 0.19f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.133f, 0.827f, 0.933f, 0.28f));
    ImGui::PushStyleColor(ImGuiCol_Border,         ImVec4(0.133f, 0.827f, 0.933f, 0.25f));
    ImGui::PushStyleColor(ImGuiCol_Text,           ImVec4(0.404f, 0.910f, 0.976f, 1.0f)); // #67e8f9
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(5);
    return clicked;
}

// Helper: primary button (web .btn.primary)
// bg: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(34,211,238,0.12))
// border: rgba(14,165,233,0.30)  color: #38bdf8
static bool primaryButton(const char* label, ImVec2 size, bool active = false) {
    if (active) {
        // web .btn.primary.active: muted, like a normal btn
        ImGui::PushStyleColor(ImGuiCol_Button,        COL_SURFACE);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(1.0f, 1.0f, 1.0f, 0.06f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(1.0f, 1.0f, 1.0f, 0.10f));
        ImGui::PushStyleColor(ImGuiCol_Border,         COL_BORDER);
        ImGui::PushStyleColor(ImGuiCol_Text,           COL_MUTED2);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.055f, 0.647f, 0.914f, 0.15f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.055f, 0.647f, 0.914f, 0.23f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.133f, 0.827f, 0.933f, 0.30f));
        ImGui::PushStyleColor(ImGuiCol_Border,         ImVec4(0.055f, 0.647f, 0.914f, 0.30f));
        ImGui::PushStyleColor(ImGuiCol_Text,           ImVec4(0.220f, 0.741f, 0.973f, 1.0f)); // #38bdf8
    }
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(5);
    return clicked;
}

// Helper: color mode button (web .color-btn)
// Per-mode gradients from CSS
static bool colorButton(const char* label, int mode, bool active, ImVec2 size) {
    // Gradient colors per mode from web CSS
    static const ImVec4 gradColors[] = {
        ImVec4(0.055f, 0.647f, 0.914f, 1.0f),  // mode 0: Deep   #0ea5e9
        ImVec4(0.133f, 0.827f, 0.933f, 1.0f),  // mode 1: Tropic #22d3ee
        ImVec4(0.976f, 0.451f, 0.086f, 1.0f),  // mode 2: Magma  #f97316
        ImVec4(0.063f, 0.725f, 0.506f, 1.0f),  // mode 3: Void   #10b981
    };
    ImVec4 gc = gradColors[mode];

    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button,    ImVec4(gc.x, gc.y, gc.z, 0.18f));
        ImGui::PushStyleColor(ImGuiCol_Border,     ImVec4(0.133f, 0.827f, 0.933f, 0.35f));
        ImGui::PushStyleColor(ImGuiCol_Text,       COL_TEXT);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,    COL_SURFACE);
        ImGui::PushStyleColor(ImGuiCol_Border,     COL_BORDER);
        ImGui::PushStyleColor(ImGuiCol_Text,       COL_MUTED2);
    }
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(gc.x, gc.y, gc.z, 0.12f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(gc.x, gc.y, gc.z, 0.22f));

    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(5);
    return clicked;
}

void draw(SPHSystem& sph, App& app) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    float panelW = 268.0f;  // match web width: 268px

    // Panel on RIGHT side — match web: top:22px, right:22px
    ImGui::SetNextWindowPos(ImVec2(displaySize.x - panelW - 22, 22), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, 0));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar;
    ImGui::Begin("##vortex", nullptr, flags);

    float gridW = ImGui::GetContentRegionAvail().x;
    float halfW = (gridW - 5.0f) * 0.5f;

    // ---- HEADER ---- match web .ui-header
    // Title: "VORTEX 3D" with accent color (can't do gradient in ImGui, use accent)
    ImGui::PushStyleColor(ImGuiCol_Text, COL_ACCENT);
    ImGui::SetWindowFontScale(1.15f);
    ImGui::Text("VORTEX 3D");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();

    // Subtitle
    ImGui::TextColored(COL_MUTED2, "SPH | CUDA GPU | %d particles", sph.params.activeN);

    // ---- DIVIDER ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ---- PARTICLES SLIDER ----
    {
        ImGui::TextColored(COL_MUTED2, "PARTICLES");
        ImGui::SameLine(gridW - 50);
        ImGui::TextColored(COL_TEXT, "%d", app.targetN);
        ImGui::Spacing();

        // Style the slider to match web range inputs
        ImGui::PushStyleColor(ImGuiCol_SliderGrab,       ImVec4(0.886f, 0.918f, 0.949f, 1.0f)); // white thumb
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive,  COL_ACCENT);
        ImGui::PushStyleColor(ImGuiCol_FrameBg,           COL_BORDER);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered,    ImVec4(1.0f, 1.0f, 1.0f, 0.10f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive,     ImVec4(0.133f, 0.827f, 0.933f, 0.20f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 3));
        ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 13.0f);

        static int sliderN = 750;
        sliderN = app.targetN;
        if (ImGui::SliderInt("##particles", &sliderN, 20, 20000, "", ImGuiSliderFlags_AlwaysClamp)) {
            if (sliderN != app.targetN) {
                app.changeParticleCount(sliderN);
            }
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(5);
    }

    // ---- DIVIDER ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ---- COLOUR LABEL ---- match web .control-label
    ImGui::TextColored(COL_MUTED2, "COLOUR");
    ImGui::Spacing();

    // ---- COLOR BUTTONS ---- 2x2 grid, match web .color-modes
    // Row 1: Deep + Tropic
    if (colorButton("Deep", 0, sph.params.colorMode == 0, ImVec2(halfW, 0)))
        sph.params.colorMode = 0;
    ImGui::SameLine(0, 5);
    if (colorButton("Tropic", 1, sph.params.colorMode == 1, ImVec2(halfW, 0)))
        sph.params.colorMode = 1;

    // Row 2: Magma + Void
    if (colorButton("Magma", 2, sph.params.colorMode == 2, ImVec2(halfW, 0)))
        sph.params.colorMode = 2;
    ImGui::SameLine(0, 5);
    if (colorButton("Void", 3, sph.params.colorMode == 3, ImVec2(halfW, 0)))
        sph.params.colorMode = 3;

    // ---- DIVIDER ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ---- ACTION BUTTONS ----
    float fullW = gridW;

    // Row: Pause + Reset (grid 2 cols) — match web .action-btns
    // Pause is "primary" style, gets "active" look when paused
    if (primaryButton(app.paused ? "RESUME" : "PAUSE", ImVec2(halfW, 0), app.paused))
        app.paused = !app.paused;
    ImGui::SameLine(0, 5);
    // Reset is default .btn style
    if (ImGui::Button("RESET", ImVec2(halfW, 0)))
        app.resetSim();

    // Tidal buttons — full width, match web .btn.tidal
    if (tidalButton("POUR          P", ImVec2(fullW, 0)))
        app.startPour();

    if (tidalButton("DRAIN         D", ImVec2(fullW, 0)))
        app.toggleDrain();

    if (tidalButton("SHAKE     SPACE", ImVec2(fullW, 0)))
        app.triggerShake();

    if (tidalButton("WAVE          W", ImVec2(fullW, 0)))
        app.triggerWave();

    ImGui::End();

    // ---- FPS BADGE ---- match web: bottom:20px, right:22px
    {
        ImGui::SetNextWindowPos(ImVec2(displaySize.x - 90, displaySize.y - 42));
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGuiWindowFlags f = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
            | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
            | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(11, 5));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.004f, 0.024f, 0.071f, 0.70f));
        ImGui::Begin("##fps", nullptr, f);
        ImGui::TextColored(COL_MUTED2, "%.0f fps", ImGui::GetIO().Framerate);
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    // ---- CONTROLS HINT ---- match web: bottom:20px, center
    {
        const char* hint = "LEFT DRAG - PUSH  |  RIGHT DRAG - ROTATE  |  ARROWS - TILT  |  SPACE - SHAKE";
        ImVec2 textSize = ImGui::CalcTextSize(hint);
        float hintW = textSize.x + 32;
        ImGui::SetNextWindowPos(ImVec2((displaySize.x - hintW) * 0.5f, displaySize.y - 42));
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGuiWindowFlags f = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
            | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
            | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16, 6));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.004f, 0.024f, 0.071f, 0.70f));
        ImGui::Begin("##hint", nullptr, f);
        ImGui::TextColored(COL_MUTED2, "%s", hint);
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

} // namespace UI
