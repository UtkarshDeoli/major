import Link from "next/link";
import Container from "@/components/global/container";

const Footer = () => {
    return (
        <footer className="flex flex-col relative items-center justify-center border-t border-foreground/5 pt-16 pb-8 px-6 lg:px-8 w-full max-w-6xl mx-auto lg:pt-32">
            <div className="grid gap-8 xl:grid-cols-3 xl:gap-8 w-full">
                <Container>
                    <div className="flex flex-col items-start justify-start md:max-w-[200px]">
                        <div className="flex items-center gap-2">
                            <img src="/logo.png" alt="Orbit" className="w-6 h-6 object-contain" />
                            <span className="text-base md:text-lg font-medium text-foreground">
                                Orbit
                            </span>
                        </div>
                        <p className="text-muted-foreground mt-4 text-sm text-start">
                            AI-powered platform that transforms your study workflow in seconds.
                        </p>
                    </div>
                </Container>

                <div className="grid-cols-2 gap-8 grid mt-16 xl:col-span-2 xl:mt-0">
                    <div className="md:grid md:grid-cols-2 md:gap-8">
                        <Container delay={0.1} className="h-auto">
                            <h3 className="text-base font-medium text-foreground">
                                Product
                            </h3>
                            <ul className="mt-4 text-sm text-muted-foreground space-y-4">
                                <li className="mt-2">
                                    <Link href="#features" className="link hover:text-foreground transition-all duration-300">
                                        Features
                                    </Link>
                                </li>
                                <li className="mt-2">
                                    <Link href="#pricing" className="link hover:text-foreground transition-all duration-300">
                                        Pricing
                                    </Link>
                                </li>
                                <li className="mt-2">
                                    <Link href="#how-it-works" className="link hover:text-foreground transition-all duration-300">
                                        How It Works
                                    </Link>
                                </li>
                                <li className="mt-2">
                                    <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                        Supported Languages
                                    </Link>
                                </li>
                            </ul>
                        </Container>
                        <Container delay={0.2} className="h-auto">
                            <div className="mt-10 md:mt-0 flex flex-col">
                                <h3 className="text-base font-medium text-foreground">
                                    Solutions
                                </h3>
                                <ul className="mt-4 text-sm text-muted-foreground space-y-4">
                                    <li>
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            Engineering Students
                                        </Link>
                                    </li>
                                    <li className="mt-2">
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            Medical Students
                                        </Link>
                                    </li>
                                    <li className="mt-2">
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            Commerce Students
                                        </Link>
                                    </li>
                                    <li className="mt-2">
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            Law Students
                                        </Link>
                                    </li>
                                </ul>
                            </div>
                        </Container>
                    </div>
                    <div className="md:grid md:grid-cols-2 md:gap-8">
                        <Container delay={0.3} className="h-auto">
                            <h3 className="text-base font-medium text-foreground">
                                Resources
                            </h3>
                            <ul className="mt-4 text-sm text-muted-foreground space-y-4">
                                <li className="mt-2">
                                    <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                        Blog
                                    </Link>
                                </li>
                                <li className="mt-2">
                                    <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                        Study Guides
                                    </Link>
                                </li>
                                <li className="mt-2">
                                    <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                        Support
                                    </Link>
                                </li>
                            </ul>
                        </Container>
                        <Container delay={0.4} className="h-auto">
                            <div className="mt-10 md:mt-0 flex flex-col">
                                <h3 className="text-base font-medium text-foreground">
                                    Company
                                </h3>
                                <ul className="mt-4 text-sm text-muted-foreground space-y-4">
                                    <li>
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            About Us
                                        </Link>
                                    </li>
                                    <li className="mt-2">
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            Privacy Policy
                                        </Link>
                                    </li>
                                    <li className="mt-2">
                                        <Link href="#" className="link hover:text-foreground transition-all duration-300">
                                            Terms & Conditions
                                        </Link>
                                    </li>
                                </ul>
                            </div>
                        </Container>
                    </div>
                </div>
            </div>

            <Container delay={0.5} className="w-full relative mt-12 lg:mt-20">
                <div className="mt-8 md:flex md:items-center justify-center footer w-full">
                    <p className="text-sm text-muted-foreground mt-8 md:mt-0">
                        &copy; {new Date().getFullYear()} Orbit. All rights reserved.
                    </p>
                </div>
            </Container>
        </footer>
    )
};

export default Footer
