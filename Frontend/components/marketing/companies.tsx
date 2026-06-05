import Container from "@/components/global/container";
import Icons from "@/components/global/icons";

const Companies = () => {
    return (
        <div className="relative flex flex-col items-center justify-center w-full py-20 mt-16 companies overflow-hidden">
            {/* Thin horizontal light line at top edge */}
            <div
                className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none"
                style={{
                    height: "1px",
                    width: "60%",
                    background: "linear-gradient(90deg, transparent 0%, rgba(85,145,243,0.35) 50%, transparent 100%)",
                }}
            />
            {/* Core bright bloom */}
            <div
                className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none"
                style={{
                    height: "20px",
                    width: "70%",
                    borderRadius: "100%",
                    filter: "blur(6rem)",
                    background: "radial-gradient(ellipse 50% 100% at 50% 50%, rgba(85,145,243,0.75) 0%, transparent 85%)",
                }}
            />
            {/* Wide soft outer halo for visible ambient glow */}
            <div
                className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none"
                style={{
                    height: "80px",
                    width: "90%",
                    borderRadius: "100%",
                    filter: "blur(8rem)",
                    background: "radial-gradient(ellipse 50% 100% at 50% 50%, rgba(85,145,243,0.35) 0%, transparent 80%)",
                }}
            />
            <Container>
                <div className="flex flex-col items-center justify-center">
                    <h4 className="text-2xl lg:text-4xl font-medium">
                        Fueling Your Success Across <span className="font-subheading italic">Every</span> Exam
                    </h4>
                </div>
            </Container>

            <Container delay={0.1}>
                <div className="flex flex-row flex-wrap items-center justify-center gap-x-8 gap-y-6 max-w-3xl mx-auto pt-16 transition-all">
                    {["JEE", "NEET", "UPSC", "CAT", "GATE", "BITSAT", "CLAT", "CA"].map((exam) => (
                        <span
                            key={exam}
                            className="text-sm md:text-base font-medium text-muted-foreground hover:text-foreground transition-colors duration-300 cursor-default"
                        >
                            {exam}
                        </span>
                    ))}
                </div>
            </Container>
        </div>
    )
};

export default Companies
