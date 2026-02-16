"""
ClimaHealth Command - Household Emergency Response System
=========================================================
Provides personalized emergency action sequences based on disease risk
and household financial situation.
"""

import gradio as gr
import numpy as np
from datetime import datetime, timedelta


class HouseholdCommand:
    def __init__(self):
        self.tasks = {
            "malaria": [
                {
                    "priority": 1,
                    "title": "Community Clinic Early Triage",
                    "description": "Visit the local district health post today for prophylactic treatment.",
                    "actions": [
                        "View Clinic Map & Wait Times",
                        "Pre-fill Triage Form",
                        "Notify Household Contacts",
                    ],
                    "cost_saved": 800,
                    "friction_score": 4,
                },
                {
                    "priority": 1,
                    "title": "Temporary Unemployment Filing",
                    "description": "File for emergency disease-related work leave within 48 hours.",
                    "actions": [
                        "Generate Filing Packet",
                        "Draft Employer Notification",
                        "Set 48h Reminder",
                    ],
                    "cost_saved": 0,
                    "friction_score": 4,
                },
                {
                    "priority": 2,
                    "title": "Emergency Food Aid (SNAP/WFP)",
                    "description": "Apply for regional food support to stabilize core expenses.",
                    "actions": [
                        "Submit Digital Application",
                        "Upload Proof of Residency",
                        "Check Application Status",
                    ],
                    "cost_saved": 240,
                    "friction_score": 4,
                },
            ],
            "cholera": [
                {
                    "priority": 1,
                    "title": "Water Source Verification",
                    "description": "Test household water supply and secure safe drinking water immediately.",
                    "actions": [
                        "Request Water Testing Kit",
                        "Locate Water Distribution Points",
                        "Order Water Purification Tablets",
                    ],
                    "cost_saved": 600,
                    "friction_score": 3,
                },
                {
                    "priority": 1,
                    "title": "ORS Pre-positioning",
                    "description": "Stock oral rehydration salts at home before outbreak escalates.",
                    "actions": [
                        "View Nearest Pharmacy",
                        "Pre-order ORS Supply",
                        "Learn Preparation Protocol",
                    ],
                    "cost_saved": 450,
                    "friction_score": 2,
                },
                {
                    "priority": 2,
                    "title": "Emergency Food Aid (SNAP/WFP)",
                    "description": "Apply for regional food support to stabilize core expenses.",
                    "actions": [
                        "Submit Digital Application",
                        "Upload Proof of Residency",
                        "Check Application Status",
                    ],
                    "cost_saved": 240,
                    "friction_score": 4,
                },
            ],
            "dengue": [
                {
                    "priority": 1,
                    "title": "Mosquito Net Distribution",
                    "description": "Obtain insecticide-treated nets for all sleeping areas.",
                    "actions": [
                        "Check Eligibility",
                        "Reserve Nets at Health Post",
                        "Schedule Pickup",
                    ],
                    "cost_saved": 350,
                    "friction_score": 2,
                },
                {
                    "priority": 1,
                    "title": "Fever Monitoring Setup",
                    "description": "Get thermometer and establish daily temperature checking routine.",
                    "actions": [
                        "Request Free Thermometer",
                        "Download Symptom Tracker App",
                        "Set Daily Reminders",
                    ],
                    "cost_saved": 200,
                    "friction_score": 1,
                },
                {
                    "priority": 2,
                    "title": "Emergency Food Aid (SNAP/WFP)",
                    "description": "Apply for regional food support to stabilize core expenses.",
                    "actions": [
                        "Submit Digital Application",
                        "Upload Proof of Residency",
                        "Check Application Status",
                    ],
                    "cost_saved": 240,
                    "friction_score": 4,
                },
            ],
        }

    def calculate_runway(self, liquid_assets, daily_revenue, health_status="normal"):
        """Calculate household financial runway in days"""
        daily_expense = daily_revenue * 0.85  # Assume 85% of revenue goes to expenses

        if health_status == "critical":
            daily_expense *= 1.5  # Medical emergency increases expenses
            daily_revenue *= 0.5  # Reduced work capacity

        net_daily = daily_revenue - daily_expense

        if net_daily >= 0:
            return 9999  # Infinite runway
        else:
            runway = liquid_assets / abs(net_daily)
            return int(runway)

    def generate_action_plan(
        self,
        region_name,
        disease,
        risk_score,
        liquid_assets,
        daily_revenue,
        health_status,
    ):
        """Generate personalized action sequence"""

        # Calculate financial parameters
        runway_days = self.calculate_runway(liquid_assets, daily_revenue, health_status)
        optimized_runway = runway_days
        total_savings = 0

        # Get disease-specific tasks
        disease_key = disease.lower()
        tasks = self.tasks.get(disease_key, self.tasks["malaria"])

        # Calculate potential savings and runway extension
        for task in tasks:
            if task["priority"] == 1:
                total_savings += task["cost_saved"]

        if runway_days < 9999:
            daily_burn = (
                liquid_assets / runway_days if runway_days > 0 else daily_revenue * 0.5
            )
            if daily_burn > 0:
                optimized_runway = int((liquid_assets + total_savings) / daily_burn)

        # Emergency mode if runway < 5 days
        emergency_mode = runway_days < 5

        # Calculate liquidity gap
        target_runway = 11  # days
        if runway_days < target_runway:
            gap_days = target_runway - runway_days
            liquidity_gap = int(gap_days * daily_revenue * 0.85)
        else:
            gap_days = 0
            liquidity_gap = 0

        # Generate AI sequencing message
        if emergency_mode:
            ai_message = f"Based on your {runway_days}-day liquidity runway, missing {gap_days} days of work will create a ${liquidity_gap} gap. The fastest way to reduce this gap is seeking immediate clinical diagnosis and treatment to prevent severe {disease} progression."
        else:
            ai_message = f"Your household has a {runway_days}-day financial buffer. Proactive engagement with Priority 1 tasks can extend this to {optimized_runway} days while preventing disease escalation."

        return {
            "emergency_mode": emergency_mode,
            "runway_default": runway_days,
            "runway_optimized": optimized_runway,
            "gap_days": gap_days,
            "liquidity_gap": liquidity_gap,
            "unmitigated_loss": (
                -total_savings if emergency_mode else -int(total_savings * 0.4)
            ),
            "aid_captured": 0,
            "max_potential": optimized_runway - runway_days,
            "ai_message": ai_message,
            "tasks": tasks,
            "health_status": health_status,
            "fragility_index": (
                min(100, int((1 - runway_days / 30) * 100)) if runway_days < 30 else 0
            ),
            "outbreak_risk": risk_score,
        }


def create_command_interface(ensemble_model, regions_config):
    """Create the ClimaHealth Command Gradio interface"""

    command_system = HouseholdCommand()

    BG = "#0a1628"

    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    .gradio-container {
        max-width: 1400px !important;
        background: #050b18 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .command-header {
        background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.15));
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .emergency-alert {
        background: rgba(220,38,38,0.12);
        border: 2px solid #DC2626;
        border-left: 6px solid #DC2626;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    .task-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .task-card:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(139,92,246,0.4);
    }
    .priority-1 {
        border-left: 4px solid #DC2626;
    }
    .priority-2 {
        border-left: 4px solid #F59E0B;
    }
    .stat-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .action-btn {
        background: rgba(139,92,246,0.15) !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        color: #A78BFA !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
        margin: 4px !important;
    }
    .action-btn:hover {
        background: rgba(139,92,246,0.25) !important;
    }
    """

    def generate_command_ui(region_name, liquid_assets, daily_revenue, health_status):
        """Generate the command center UI"""

        info = regions_config[region_name]
        disease = info["disease_name"]

        # Get risk assessment from ensemble
        # For demo, use mock risk score - in production, fetch from ensemble
        risk_score = np.random.randint(60, 85)

        plan = command_system.generate_action_plan(
            region_name,
            disease,
            risk_score,
            liquid_assets,
            daily_revenue,
            health_status,
        )

        # Build HTML output
        html = f"""
        <div style="background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(59,130,246,0.08)); 
                    border: 1px solid rgba(139,92,246,0.2); border-radius: 16px; padding: 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #8B5CF6, #3B82F6);
                            border-radius: 12px; display: flex; align-items: center; justify-content: center;
                            font-size: 24px;">⚡</div>
                <div>
                    <div style="font-size: 24px; font-weight: 700; color: white; font-family: 'Inter', sans-serif;">
                        ClimaHealth<span style="color: #8B5CF6;">Command</span>
                    </div>
                    <div style="font-size: 11px; color: #8899aa; letter-spacing: 1.5px; text-transform: uppercase;
                                font-family: 'JetBrains Mono', monospace;">
                        Household Emergency Response System
                    </div>
                </div>
            </div>
        </div>
        """

        # Emergency alert if needed
        if plan["emergency_mode"]:
            html += f"""
            <div style="background: rgba(220,38,38,0.12); border: 2px solid #DC2626; border-left: 6px solid #DC2626;
                        border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 32px;">⚠️</div>
                    <div style="flex: 1;">
                        <div style="font-size: 13px; font-weight: 700; color: #FCA5A5; letter-spacing: 1px;
                                    font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;">
                            ⚠ EMERGENCY MODE ACTIVE
                        </div>
                        <div style="font-size: 12px; color: rgba(255,255,255,0.85); line-height: 1.6;">
                            Priority 1 tasks must be initiated within 24 hours to prevent liquidity collapse.
                        </div>
                    </div>
                </div>
            </div>
            """

        # AI Action Sequencing
        html += f"""
        <div style="background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2);
                    border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="font-size: 11px; color: #60A5FA; font-weight: 700; letter-spacing: 1.5px;
                        font-family: 'JetBrains Mono', monospace; margin-bottom: 12px;">
                ⚡ AI ACTION SEQUENCING
            </div>
            <div style="font-size: 15px; color: white; line-height: 1.7; font-weight: 500;">
                {plan["ai_message"]}
            </div>
        </div>
        """

        # Target area info
        html += f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 12px; padding: 16px; margin-bottom: 20px;">
            <div style="font-size: 11px; color: #8899aa; font-family: 'JetBrains Mono', monospace;
                        letter-spacing: 1px; margin-bottom: 8px;">TARGET BIO-THREAT AREA</div>
            <div style="font-size: 20px; font-weight: 700; color: white;">
                {region_name.split(' (')[0]}
            </div>
            <div style="font-size: 12px; color: #8899aa; margin-top: 4px;">
                {disease} Risk Zone • Local Outbreak: {plan["outbreak_risk"]}%
            </div>
        </div>
        """

        # Household parameters
        html += f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 12px; padding: 16px; margin-bottom: 20px;">
            <div style="font-size: 11px; color: #8899aa; font-family: 'JetBrains Mono', monospace;
                        letter-spacing: 1px; margin-bottom: 12px;">HOUSEHOLD FRAGILITY PARAMETERS</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div>
                    <div style="font-size: 10px; color: #667788; font-family: 'JetBrains Mono', monospace;">
                        LIQUID ASSETS ($)
                    </div>
                    <div style="font-size: 22px; font-weight: 700; color: white;">${liquid_assets}</div>
                </div>
                <div>
                    <div style="font-size: 10px; color: #667788; font-family: 'JetBrains Mono', monospace;">
                        DAILY REVENUE ($)
                    </div>
                    <div style="font-size: 22px; font-weight: 700; color: white;">${daily_revenue}</div>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08);">
                <div style="font-size: 10px; color: #667788; font-family: 'JetBrains Mono', monospace;
                            margin-bottom: 6px;">HEALTH STATUS</div>
                <div style="display: inline-block; padding: 6px 12px; border-radius: 6px;
                            background: {'rgba(220,38,38,0.15)' if health_status == 'critical' else 'rgba(16,185,129,0.15)'};
                            border: 1px solid {'#DC2626' if health_status == 'critical' else '#10B981'};">
                    <span style="font-size: 11px; font-weight: 700; 
                                 color: {'#FCA5A5' if health_status == 'critical' else '#6EE7B7'};
                                 font-family: 'JetBrains Mono', monospace;">
                        {'🔴 CRITICAL SHOCK' if health_status == 'critical' else '🟢 NORMAL'}
                    </span>
                </div>
            </div>
        </div>
        """

        # Risk telemetry
        html += f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; padding: 16px; margin-bottom: 20px;">
            <div style="font-size: 11px; color: #8899aa; font-family: 'JetBrains Mono', monospace;
                        letter-spacing: 1px; margin-bottom: 12px;">📡 RISK TELEMETRY</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div style="text-align: center; padding: 12px; background: rgba(220,38,38,0.08);
                            border: 1px solid rgba(220,38,38,0.2); border-radius: 8px;">
                    <div style="font-size: 10px; color: #FCA5A5; font-family: 'JetBrains Mono', monospace;">
                        LOCAL OUTBREAK
                    </div>
                    <div style="font-size: 28px; font-weight: 700; color: #EF4444; margin: 4px 0;">
                        {plan["outbreak_risk"]}%
                    </div>
                </div>
                <div style="text-align: center; padding: 12px; background: rgba(245,158,11,0.08);
                            border: 1px solid rgba(245,158,11,0.2); border-radius: 8px;">
                    <div style="font-size: 10px; color: #FDE68A; font-family: 'JetBrains Mono', monospace;">
                        FRAGILITY INDEX
                    </div>
                    <div style="font-size: 28px; font-weight: 700; color: #F59E0B; margin: 4px 0;">
                        {plan["fragility_index"]}
                    </div>
                </div>
            </div>
        </div>
        """

        return html

    with gr.Blocks(
        title="ClimaHealth Command", css=CSS, theme=gr.themes.Base()
    ) as interface:

        gr.Markdown("# ⚡ ClimaHealth Command Center")
        gr.Markdown("### Execution Command Center")
        gr.Markdown("**Impact / (Time × Friction)**")

        with gr.Row():
            with gr.Column(scale=1):
                region_select = gr.Dropdown(
                    choices=list(regions_config.keys()),
                    value="Nairobi, Kenya (Malaria)",
                    label="TARGET BIO-THREAT AREA",
                )

                gr.Markdown("### Household Fragility Parameters")
                liquid_assets = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=140,
                    step=10,
                    label="Liquid Assets ($)",
                )
                daily_revenue = gr.Slider(
                    minimum=50,
                    maximum=300,
                    value=130,
                    step=10,
                    label="Daily Revenue ($)",
                )
                health_toggle = gr.Radio(
                    choices=["normal", "critical"],
                    value="critical",
                    label="Health Status",
                )

                generate_btn = gr.Button("🎯 Generate Action Plan", variant="primary")

            with gr.Column(scale=2):
                output_html = gr.HTML()

        generate_btn.click(
            fn=generate_command_ui,
            inputs=[region_select, liquid_assets, daily_revenue, health_toggle],
            outputs=[output_html],
        )

        # Load default on startup
        interface.load(
            fn=generate_command_ui,
            inputs=[region_select, liquid_assets, daily_revenue, health_toggle],
            outputs=[output_html],
        )

    return interface
