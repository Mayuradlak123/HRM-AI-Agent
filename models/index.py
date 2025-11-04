# db/models.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Literal
from datetime import date, datetime
from enum import Enum
import uuid
# Add this to your db/models.py
class BankDetails(BaseModel):
    bank_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bank_name: str
    bank_address: str
    ifsc_code: str
    account_number: str
    account_name: str
    account_type: str
    account_balance: float
    account_currency: str = "INR"
    account_status: bool = True
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
class CompanyInfo(BaseModel):
    company_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    legal_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    founded_year: Optional[int] = None
    
    # Contact Information
    headquarters_address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    
    # Business Information
    employee_count: Optional[int] = None
    annual_revenue: Optional[float] = None
    revenue_currency: str = "INR"
    
    # Locations
    locations: List[str] = Field(default_factory=list)
    
    # Leadership
    ceo_name: Optional[str] = None
    founded_by: Optional[List[str]] = Field(default_factory=list)
    
    # Policies & Values
    mission_statement: Optional[str] = None
    vision_statement: Optional[str] = None
    core_values: List[str] = Field(default_factory=list)
    
    # Work Culture
    work_culture: Optional[str] = None
    benefits_overview: Optional[str] = None
    remote_work_policy: Optional[str] = None
    
    # Social Media & Links
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    instagram_url: Optional[str] = None
    
    # Operating Information
    time_zone: str = "Asia/Kolkata"
    working_days: List[str] = Field(default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    working_hours: str = "9:00 AM - 6:00 PM"
    
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CompanyPolicy(BaseModel):
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_id: str
    policy_type: str  # "leave", "attendance", "code_of_conduct", "remote_work", etc.
    title: str
    content: str
    effective_date: date = Field(default_factory=lambda: date.today())
    version: str = "1.0"
    department_specific: Optional[str] = None  # If policy applies to specific department
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CompanyAnnouncement(BaseModel):
    announcement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_id: str
    title: str
    content: str
    announcement_type: str = "general"  # "general", "policy", "event", "celebration", "emergency"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    target_audience: str = "all"  # "all", "department_specific", "role_specific"
    department_id: Optional[str] = None
    author_id: str
    published_date: datetime = Field(default_factory=datetime.utcnow)
    expiry_date: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
class Department(BaseModel):
    department_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Designation(BaseModel):
    designation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    department_id: str
    title: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class UserRole(str, Enum):
    HR = "hr"
    EMPLOYEE = "employee"
    ADMIN = "admin"

class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    first_name: str
    last_name: str
    role: UserRole
    department_id: Optional[str] = None
    designation_id: Optional[str] = None
    manager_id: Optional[str] = None
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERN = "intern"

class Employee(BaseModel):
    employee_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    first_name: str
    last_name: str
    email: EmailStr
    department_id: str
    designation_id: str
    start_date: datetime
    manager_id: Optional[str] = None
    location: Optional[str] = None
    phone: Optional[str] = None
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    is_active: bool = True

    base_salary: Optional[float] = None
    salary_currency: str = "INR"
    grade: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LeaveType(BaseModel):
    leave_type_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str
    name: str
    annual_quota: float = 0.0
    carry_forward: bool = False
    max_balance: Optional[float] = None
    requires_approval: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LeaveBalance(BaseModel):
    balance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    leave_type_code: str
    balance: float = 0.0
    as_of: datetime = Field(default_factory=datetime.utcnow)

class LeaveStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

class LeaveRequest(BaseModel):
    leave_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    leave_type_code: str
    start_date: date
    end_date: date
    days: float
    reason: Optional[str] = None
    status: LeaveStatus = LeaveStatus.PENDING
    approver_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AttendanceRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    date: date
    check_in: Optional[datetime] = None
    check_out: Optional[datetime] = None
    work_hours: Optional[float] = None
    status: Literal["present", "absent", "leave", "holiday", "weekend"] = "present"
    notes: Optional[str] = None

class SalaryComponent(BaseModel):
    name: str
    amount: float
    taxable: bool = True

class SalaryStructure(BaseModel):
    structure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    effective_from: date
    components: List[SalaryComponent] = Field(default_factory=list)
    ctc_annual: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PayrollRecord(BaseModel):
    payroll_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    month: int
    year: int
    gross_pay: float
    net_pay: float
    components: Dict[str, float] = Field(default_factory=dict)
    deductions: Dict[str, float] = Field(default_factory=dict)
    pay_date: datetime = Field(default_factory=datetime.utcnow)
    currency: str = "INR"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BenefitEnrollment(BaseModel):
    enrollment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    benefit_name: str
    provider: Optional[str] = None
    policy_number: Optional[str] = None
    start_date: date
    end_date: Optional[date] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AssetAssignment(BaseModel):
    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    asset_type: str
    make_model: Optional[str] = None
    serial_no: Optional[str] = None
    assigned_on: date = Field(default_factory=lambda: date.today())
    returned_on: Optional[date] = None
    notes: Optional[str] = None

class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    user_message: str
    agent_response: str
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Chat"
    messages: List[ChatMessage] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class KnowledgeDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    document_type: str
    category: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding_id: Optional[str] = None
    created_by: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict),
    mistralai_response: Optional[str] = None

class DeleteChatRequest(BaseModel):
    session_id: str