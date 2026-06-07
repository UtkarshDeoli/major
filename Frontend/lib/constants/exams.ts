export interface PresetExam {
  id: string;
  name: string;
  tagline: string;
  icon: string;
  subjects: string[];
}

export const PRESET_EXAMS: PresetExam[] = [
  {
    id: "jee-mains",
    name: "JEE Mains",
    tagline: "Joint Entrance Examination for Engineering",
    icon: "atom",
    subjects: ["Physics", "Chemistry", "Mathematics"],
  },
  {
    id: "neet",
    name: "NEET",
    tagline: "National Eligibility cum Entrance Test for Medicine",
    icon: "heart-pulse",
    subjects: ["Physics", "Chemistry", "Biology"],
  },
  {
    id: "upsc",
    name: "UPSC",
    tagline: "Union Public Service Commission Civil Services",
    icon: "landmark",
    subjects: [
      "History",
      "Geography",
      "Polity",
      "Economics",
      "Environment",
      "Science & Technology",
      "Current Affairs",
    ],
  },
  {
    id: "cat",
    name: "CAT",
    tagline: "Common Admission Test for IIMs",
    icon: "bar-chart-3",
    subjects: [
      "Quantitative Aptitude",
      "Verbal Ability",
      "Logical Reasoning",
      "Data Interpretation",
    ],
  },
  {
    id: "gate",
    name: "GATE",
    tagline: "Graduate Aptitude Test in Engineering",
    icon: "cpu",
    subjects: [
      "Engineering Mathematics",
      "General Aptitude",
      "Core Subject",
    ],
  },
  {
    id: "cbse-12th",
    name: "CBSE 12th",
    tagline: "Central Board of Secondary Education Senior Secondary",
    icon: "graduation-cap",
    subjects: [
      "Physics",
      "Chemistry",
      "Mathematics",
      "Biology",
      "English",
      "Accountancy",
      "Business Studies",
      "Economics",
      "History",
      "Political Science",
    ],
  },
];

export interface IndianLanguage {
  code: string;
  name: string;
}

export const INDIAN_LANGUAGES: IndianLanguage[] = [
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "mr", name: "Marathi" },
  { code: "bn", name: "Bengali" },
  { code: "gu", name: "Gujarati" },
  { code: "kn", name: "Kannada" },
  { code: "ml", name: "Malayalam" },
  { code: "pa", name: "Punjabi" },
  { code: "ur", name: "Urdu" },
];
