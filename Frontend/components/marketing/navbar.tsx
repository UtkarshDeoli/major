import { NAV_LINKS } from "@/lib/constants/links";
import Link from "next/link";
import Wrapper from "@/components/global/wrapper";
import { Button } from "@/components/ui/button";
import MobileMenu from "./mobile-menu";

const Navbar = () => {
    return (
        <header className="sticky top-0 w-full h-16 bg-background/80 backdrop-blur-sm z-50 border-b border-border/30">
            <Wrapper className="h-full">
                <div className="flex items-center justify-between h-full">
                    <div className="flex items-center">
                        <Link href="/" className="flex items-center gap-2">
                            <img src="/logo.png" alt="Orbit" className="w-7 h-7 object-contain" />
                            <span className="text-xl font-semibold hidden lg:block">
                                Orbit
                            </span>
                        </Link>
                    </div>

                    <div className="hidden lg:flex items-center gap-4">
                        <ul className="flex items-center gap-8">
                            {NAV_LINKS.map((link, index) => (
                                <li key={index} className="text-sm font-medium link">
                                    <Link href={link.href}>
                                        {link.name}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="flex items-center gap-4">
                        <Link href="/login" className="hidden lg:block">
                            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
                                Log in
                            </Button>
                        </Link>
                        <Link href="/dashboard" className="hidden lg:block">
                            <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                                Get Started
                            </Button>
                        </Link>
                        <MobileMenu />
                    </div>
                </div>
            </Wrapper>
        </header>
    )
};

export default Navbar
